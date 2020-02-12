import {
  BufferAttribute,
  Matrix4,
  Vector3
} from 'https://raw.githack.com/mrdoob/three.js/r111/build/three.module.js';

const painterSort = (a, b) => {
  return a.z - b.z;
};

const reversePainterSort = (a, b) => {
  return b.z - a.z;
};

const ShaderLibs = {};

ShaderLibs.MeshBasicMaterial = {
  vertexShaderCode: `
  layout(set=0, binding=0) uniform Uniforms {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    mat3 normalMatrix;
  } uniforms;

  layout(location = 0) in vec3 position;
  layout(location = 1) in vec3 normal;
  layout(location = 2) in vec2 uv;

  layout(location = 0) out vec3 fragNormal;
  layout(location = 1) out vec2 fragUv;

  void main() {
    gl_Position = uniforms.projectionMatrix * uniforms.viewMatrix * uniforms.modelMatrix * vec4(position, 1.0);
    fragUv = uv;
    fragNormal = normal;
  }`,
  fragmentShaderCode: `
  layout(set=0, binding=1) uniform Uniforms {
    vec3 color;
    float opacity;
  } uniforms;

  #ifdef USE_MAP
    layout(set=0, binding=2) uniform sampler mapSampler;
    layout(set=0, binding=3) uniform texture2D map;
  #endif

  layout(location = 0) in vec3 fragNormal;
  layout(location = 1) in vec2 fragUv;
  layout(location = 0) out vec4 outColor;

  void main() {
    outColor = vec4(uniforms.color, uniforms.opacity);
    #ifdef USE_MAP
      outColor *= texture(sampler2D(map, mapSampler), fragUv);
    #endif
  }`
};

ShaderLibs.MeshNormalMaterial = {
  vertexShaderCode: `
  layout(set=0, binding=0) uniform Uniforms {
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    mat3 normalMatrix;
  } uniforms;

  layout(location = 0) in vec3 position;
  layout(location = 1) in vec3 normal;
  layout(location = 2) in vec2 uv;

  layout(location = 0) out vec3 fragNormal;
  layout(location = 1) out vec2 fragUv;

  void main() {
    gl_Position = uniforms.projectionMatrix * uniforms.viewMatrix * uniforms.modelMatrix * vec4(position, 1.0);
    fragUv = uv;
    fragNormal = normalize(uniforms.normalMatrix * normal);
  }`,
  fragmentShaderCode: `
  layout(set=0, binding=1) uniform Uniforms {
    vec3 color;
    float opacity;
  } uniforms;

  layout(location = 0) in vec3 fragNormal;
  layout(location = 1) in vec2 fragUv;
  layout(location = 0) out vec4 outColor;

  void main() {
    // "+ (normalize(fragNormal) * 0.5 + 0.5)" is to check each surface
    vec3 normal = normalize(fragNormal);
    outColor = vec4((normal * 0.5 + 0.5), uniforms.opacity);
  }`
};

export default class WebGPURenderer {
  constructor(options={}) {
    this._projScreenMatrix = new Matrix4();
    this._width = 640;
    this._height = 480;
    this._pixelRatio = 1.0;

    this._device = null;
    this._verticesManager = null;
    this._indicesManager = null;
    this._uniformsManager = null;
    this._programManager = null;
    this._swapChain = null;
    this._format = 'bgra8unorm';
    this._sampleCount = 4;
    this._passEncoder = null;
    this._commandEncoder = null;
    this._colorAttachment = null;
    this._depthStencilAttachment = null;

    this._cache = {
      currentProgram: null
    };

    this.domElement = document.createElement('canvas');
    this.context = this.domElement.getContext('gpupresent');

    if (!options.device || !options.glslang) {
      throw new Error('WebGPURenderer: the constructor must take device and glslang parameters.');
    }

    this._device = options.device;
    this._glslang = options.glslang;

    this._verticesManager = new WebGPUVerticesManager();
    this._indicesManager = new WebGPUIndicesManager();
    this._uniformsManager = new WebGPUUniformsManager();
    this._programManager = new WebGPUProgramManager(this._glslang);
    this._swapChain = this.context.configureSwapChain({
      device: this._device,
      format: this._format
    });
    this._setSize(this._width, this._height, this._pixelRatio);
  }

  setSize(width, height) {
    this._setSize(width, height, this._pixelRatio);
  }

  setPixelRatio(pixelRatio) {
    this._setSize(this._width, this._height, pixelRatio);
  }

  /**
   * @param {Scene} scene
   * @param {Camera} camera
   */
  render(scene, camera) {
    if (!this._device) {
      return;
    }

    this._cache.currentProgram = null;

    scene.updateMatrixWorld();
    if (!camera.parent) {
      camera.updateMatrixWorld();
    }

    this._projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);

    const renderList = [];
    this._projectObject(scene, camera, renderList);

    const opaqueObjects = [];
    const transparentObjects = [];
    for (const element of renderList) {
      if (element.object.material.transparent) {
        transparentObjects.push(element);
      } else {
        opaqueObjects.push(element);
      }
    }

    transparentObjects.sort(painterSort);
    opaqueObjects.sort(reversePainterSort);

    this._setup();
    this._renderObjects(opaqueObjects, camera);
    this._renderObjects(transparentObjects, camera);
    this._finalize();
  }

  _setSize(width, height, pixelRatio) {
    this._width = width;
    this._height = height;
    this._pixelRatio = pixelRatio;
    const canvas = this.domElement;
    canvas.width = Math.floor(this._width * this._pixelRatio);
    canvas.height = Math.floor(this._height * this._pixelRatio);
    canvas.style.width = this._width + 'px';
    canvas.style.height = this._height + 'px';
    if (!this._device) {
      return;
    }
    const colorTexture = this._device.createTexture({
      size: {
        width: Math.floor(this._width * this._pixelRatio),
        height: Math.floor(this._height * this._pixelRatio),
        depth: 1
      },
      sampleCount: this._sampleCount,
      format: this._format,
      usage: GPUTextureUsage.OUTPUT_ATTACHMENT
    });
    this._colorAttachment = colorTexture.createView();
    const depthStencilTexture = this._device.createTexture({
      size: {
        width: Math.floor(this._width * this._pixelRatio),
        height: Math.floor(this._height * this._pixelRatio),
        depth: 1
      },
      sampleCount: this._sampleCount,
      format: "depth24plus-stencil8",
      usage: GPUTextureUsage.OUTPUT_ATTACHMENT
    });
    this._depthStencilAttachment = depthStencilTexture.createView();

    // I don't know why but seems necessary to render... @TODO: resolve
    if (canvas.parentElement) {
      canvas.parentElement.appendChild(canvas);
    }
  }

  _projectObject(object, camera, renderList) {
    if (!object.visible) {
      return;
    }
    if (object.isMesh) {
      if (object.material.visible) {
        const vector3 = new Vector3();
        vector3.setFromMatrixPosition(object.matrixWorld)
          .applyMatrix4(this._projScreenMatrix);
        renderList.push({
          object: object,
          z: vector3.z
        });
      }
    }
    for (const child of object.children) {
      this._projectObject(child, camera, renderList);
    }
  }

  _setup() {
    this._commandEncoder = this._device.createCommandEncoder({});
  
    const renderPassDescriptor = {
      colorAttachments: [{
        attachment: this._colorAttachment,
        resolveTarget: this._swapChain.getCurrentTexture().createView(),
        loadValue: {r: 1.0, g: 1.0, b: 1.0, a: 1.0}
      }],
      depthStencilAttachment: {
        attachment: this._depthStencilAttachment,
        depthLoadValue: 1.0,
        depthStoreOp: 'store',
        stencilLoadValue: 0,
        stencilStoreOp: 'store',
      }
    };

    this._passEncoder = this._commandEncoder.beginRenderPass(renderPassDescriptor);
  }

  _renderObjects(objects, camera) {
    for (const object of objects) {
      this._renderObject(object.object, camera);
    }
  }

  _renderObject(object, camera) {
    const program = this._programManager.get(object.material, this._device);

    if (this._cache.currentProgram !== program) {
      this._passEncoder.setPipeline(program.pipeline);
      this._cache.currentProgram = program;
    }

    const vertices = this._verticesManager.get(object.geometry, program, this._device);
    const uniforms = this._uniformsManager.get(object, camera, program, this._device);
    const indexAttribute = object.geometry.getIndex();

    this._passEncoder.setVertexBuffer(0, vertices.buffer);
    this._passEncoder.setBindGroup(0, uniforms.bindGroup);

    if (indexAttribute) {
      const indices = this._indicesManager.get(object.geometry, this._device);
      this._passEncoder.setIndexBuffer(indices.buffer);
      const indexCount = indexAttribute.array.length;
      this._passEncoder.drawIndexed(indexCount, 1, 0, 0, 0);
    } else {
      const vertexCount = object.geometry.getAttribute('position').array.length;
      this._passEncoder.draw(vertexCount, 1, 0, 0);
    }
  }

  _finalize() {
    this._passEncoder.endPass();
    this._device.defaultQueue.submit([this._commandEncoder.finish()]);
  }
}

class WebGPUProgramManager {
  constructor(glslang) {
    this.glslang = glslang;
    this.map = new Map();
  }

  get(material, device) {
    const key = material.type + ':' + (material.map ? 'map' : '');
    if (!this.map.has(key)) {
      const shader = ShaderLibs[material.type];

      if (!shader) {
        throw new Error('This type of material is not supported yet. ' +  key);
      }

      const vertexShaderCode = '#version 450\n' +
        shader.vertexShaderCode;

      const fragmentShaderCode = '#version 450\n' +
        (material.map ? '#define USE_MAP\n' : '') +
        shader.fragmentShaderCode;

      this.map.set(key, new WebGPUProgram(
        this.glslang,
        material,
        vertexShaderCode,
        fragmentShaderCode,
        4,
        device
      ));
    }
    return this.map.get(key);
  }
}

class WebGPUProgram {
  constructor(glslang, material, vertexShaderCode, fragmentShaderCode, sampleCount, device) {
    const bindings = [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      type: 'uniform-buffer'
    }, {
      binding: 1,
      visibility: GPUShaderStage.FRAGMENT,
      type: 'uniform-buffer'
    }];

    if (material.map) {
      bindings.push({
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        type: 'sampler'
      });
      bindings.push({
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        type: 'sampled-texture'
      });
    }

    this.uniformGroupLayout = device.createBindGroupLayout({
      bindings: bindings,
    });

    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({bindGroupLayouts: [this.uniformGroupLayout]}),
      vertexStage: {
        module: device.createShaderModule({
          code: glslang.compileGLSL(vertexShaderCode, 'vertex'),
          source: vertexShaderCode,
          transform: source => glslang.compileGLSL(source, 'vertex')
        }),
        entryPoint: 'main'
      },
      fragmentStage: {
        module: device.createShaderModule({
          code: glslang.compileGLSL(fragmentShaderCode, 'fragment'),
          source: fragmentShaderCode,
          transform: source => glslang.compileGLSL(source, 'fragment')
        }),
        entryPoint: 'main'
      },
      primitiveTopology: 'triangle-list',
      depthStencilState: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus-stencil8'
      },
      vertexState: {
        indexFormat: 'uint16',
        vertexBuffers: [{
          arrayStride: 4 * (3 + 3 + 2),
          attributes: [{
            // position
            shaderLocation: 0,
            offset: 0,
            format: 'float3'
          }, {
            // normal
            shaderLocation: 1,
            offset: 4 * 3,
            format: 'float3'
          }, {
            // uv
            shaderLocation: 2,
            offset: 4 * (3 + 3),
            format: 'float2'
          }]
        }]
      },
      colorStates: [{
        format: 'bgra8unorm',
        colorBlend: {
          srcFactor: 'src-alpha',
          dstFactor: 'one-minus-src-alpha',
          operation: 'add'
        }
      }],
      sampleCount: sampleCount
    });
  }

  createVertices(geometry, device) {
    const position = geometry.getAttribute('position');
    const array = new Float32Array((3 + 3 + 2) * position.count);
    return new WebGPUVertices(array, device);
  }

  createUniforms(device, textures) {
    const buffers = this._createUniformBuffers(device);
    const bindGroup = this._createUniformBindGroup(buffers, textures, device);
    return new WebGPUUniforms(buffers, bindGroup);
  }

  _createUniformBuffers(device) {
    const buffers = [];
    buffers.push(new WebGPUUniformBuffer((16 * 3 + 12) * 4, device)); // vertex
    buffers.push(new WebGPUUniformBuffer((3 + 1) * 4, device)); // fragment
    return buffers;
  }

  _createUniformBindGroup(buffers, textures, device) {
    const bindings = [];
    for (let i = 0; i < buffers.length; i++) {
      const buffer = buffers[i];
      bindings.push({
        binding: i,
        resource: {
          buffer: buffer.buffer,
          size: buffer.byteLength
        }
      });
    }
    for (let i = 0; i < textures.length; i++) {
      bindings.push({
        binding: bindings.length,
        resource: textures[i].sampler.sampler
      });
      bindings.push({
        binding: bindings.length,
        resource: textures[i].texture.createView()
      });
    }
    return device.createBindGroup({
      layout: this.uniformGroupLayout,
      bindings: bindings
    });
  }
}

class WebGPUUniformsManager {
  constructor() {
    this.map = new Map();
  }

  get(object, camera, program, device) {
    if (!this.map.has(object)) {
      const textures = [];
      if (object.material.map && object.material.map.image) {
        const image = object.material.map.image;
        const texture = new WebGPUTexture(image.width, image.height, device);
        texture.upload(image, device);
        textures.push(texture);
      }
      this.map.set(object, program.createUniforms(device, textures));
    }
    const uniforms = this.map.get(object);
    uniforms.update(object, camera);
    return uniforms;
  }
}

class WebGPUUniforms {
  constructor(buffers, bindGroup) {
    this.buffers = buffers;
    this.bindGroup = bindGroup;
  }

  update(object, camera) {
    object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
    object.normalMatrix.getNormalMatrix(object.modelViewMatrix);

    const material = object.material;
    this.buffers[0].updateMatrix4(0, object.matrixWorld);
    this.buffers[0].updateMatrix4(16 * 4, camera.matrixWorldInverse);
    this.buffers[0].updateMatrix4(16 * 2 * 4, camera.projectionMatrix);
    this.buffers[0].updateMatrix3(16 * 3 * 4, object.normalMatrix);
    this.buffers[0].upload();

    if (material.color) {
      this.buffers[1].updateColor(0, material.color);
    }
    this.buffers[1].updateFloat(3 * 4, material.opacity);
    this.buffers[1].upload();
  }
}

class WebGPUUniformBuffer {
  constructor(byteLength, device) {
    this.buffer = device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.byteLength = byteLength;
    const arrayBuffer = new ArrayBuffer(byteLength);
    this.float32Array = new Float32Array(arrayBuffer);
  }

  upload() {
    this.buffer.setSubData(0, this.float32Array);
  }

  updateMatrix4(offset, matrix4) {
    const index = offset / 4;
    for (let i = 0; i < 16; i++) {
      this.float32Array[index + i] = matrix4.elements[i];
    }
  }

  updateMatrix3(offset, matrix3) {
    const index = offset / 4;
    this.float32Array[index + 0] = matrix3.elements[0];
    this.float32Array[index + 1] = matrix3.elements[1];
    this.float32Array[index + 2] = matrix3.elements[2];
    this.float32Array[index + 4] = matrix3.elements[3];
    this.float32Array[index + 5] = matrix3.elements[4];
    this.float32Array[index + 6] = matrix3.elements[5];
    this.float32Array[index + 8] = matrix3.elements[6];
    this.float32Array[index + 9] = matrix3.elements[7];
    this.float32Array[index + 10] = matrix3.elements[8];
  }

  updateVector3(offset, vec3) {
    const index = offset / 4;
    this.float32Array[index] = vec3.x;
    this.float32Array[index + 1] = vec3.y;
    this.float32Array[index + 2] = vec3.z;
  }

  updateColor(offset, color) {
    const index = offset / 4;
    this.float32Array[index] = color.r;
    this.float32Array[index + 1] = color.g;
    this.float32Array[index + 2] = color.b;
  }

  updateFloat(offset, value) {
    this.float32Array[offset / 4] = value;
  }
}

class WebGPUUniformSampler {
  constructor(device) {
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear'
    });
  }
}

class WebGPUTexture {
  constructor(width, height, device) {
    this.width = width;
    this.height = height;
    this.sampler = new WebGPUUniformSampler(device);

    this.texture = device.createTexture({
      size: {
        width: this.width,
        height: this.height,
        depth: 1
      },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.SAMPLED | GPUTextureUsage.COPY_DST
    });

    this.buffer = device.createBuffer({
      size: this.width * this.height * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
  }

  upload(image, device) {
    const canvas = document.createElement('canvas');
    canvas.width = this.width;
    canvas.height = this.height;
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);
    const imageData = context.getImageData(0, 0, this.width, this.height);
    this.buffer.setSubData(0, imageData.data);

    const encoder = device.createCommandEncoder({});
    encoder.copyBufferToTexture({
      buffer: this.buffer,
      rowPitch: this.width * 4,
      imageHeight: 0
    }, {
      texture: this.texture
    }, {
      width: this.width,
      height: this.height,
      depth: 1
    });

    device.defaultQueue.submit([encoder.finish()]);
  }
}

class WebGPUVerticesManager {
  constructor() {
    this.map = new Map();
  }

  get(geometry, program, device) {
    if (!this.map.has(geometry)) {
      const vertices = program.createVertices(geometry, device);
      vertices.update(geometry);
      vertices.upload();
      this.map.set(geometry, vertices);
    }
    return this.map.get(geometry);
  }
}

class WebGPUVertices {
  constructor(array, device) {
    this.array = array;
    this.buffer = device.createBuffer({
      size: this.array.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
  }

  update(geometry) {
    const position = geometry.getAttribute('position');
    const normal = geometry.getAttribute('normal');

    // @TODO: Remove this temporal workaround
    if (!geometry.getAttribute('uv')) {
      geometry.setAttribute('uv', new BufferAttribute(new Float32Array(position.count * 2), 2));
    }
    const uv = geometry.getAttribute('uv');

    let dataIndex = 0;
    let positionIndex = 0;
    let normalIndex = 0;
    let uvIndex = 0;
    for (let i = 0; i < position.count; i++) {
      for (let j = 0; j < position.itemSize; j++) {
        this.array[dataIndex++] = position.array[positionIndex++];
      }
      for (let j = 0; j < normal.itemSize; j++) {
        this.array[dataIndex++] = normal.array[normalIndex++];
      }
      for (let j = 0; j < uv.itemSize; j++) {
        this.array[dataIndex++] = uv.array[uvIndex++];
      }
    }
  }

  upload() {
    this.buffer.setSubData(0, this.array);
  }
}

class WebGPUIndicesManager {
  constructor() {
    this.map = new Map();
  }

  get(geometry, device) {
    if (!this.map.has(geometry)) {
      const indices = new WebGPUIndices(
        // Buffer subdata size must be a multiple of 4 bytes
        new Uint16Array(Math.floor((geometry.getIndex().array.byteLength + 3) / 4) * 4),
        device
      );
      indices.update(geometry);
      indices.upload();
      this.map.set(geometry, indices);
    }
    return this.map.get(geometry);
  }
}

class WebGPUIndices {
  constructor(array, device) {
    this.array = array;
    this.buffer = device.createBuffer({
      size: this.array.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
  }

  update(geometry) {
    const index = geometry.getIndex();
    for (let i = 0; i < index.array.length; i++) {
      this.array[i] = index.array[i];
    }
  }

  upload() {
    this.buffer.setSubData(0, this.array);
  }
}
