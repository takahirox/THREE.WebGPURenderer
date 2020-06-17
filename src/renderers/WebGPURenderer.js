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

const defaultAttributeDefinitions = [{
  name: 'position',
  format: 'float3'
}, {
  name: 'normal',
  format: 'float3'
}, {
  name: 'uv',
  format: 'float2'
}];

const defaultVertexUniformsDefinitions = [{
  name: 'modelMatrix',
  format: 'matrix4'
}, {
  name: 'viewMatrix',
  format: 'matrix4'
}, {
  name: 'projectionMatrix',
  format: 'matrix4'
}, {
  name: 'normalMatrix',
  format: 'matrix3'
}];

const defaultFragmentUniformsDefinitions = [{
  name: 'diffuse',
  format: 'color'
}, {
  name: 'opacity',
  format: 'float'
}];

const offsetTable = {
  'float': 4,
  'float2': 8,
  'float3': 12,
  'color': 12,
  'float4': 16,
  'matrix3': 12 * 4,
  'matrix4': 16 * 4
};

const ShaderLibs = {};

// PBR shader code is based on https://github.com/KhronosGroup/glTF-Sample-Viewer/
ShaderLibs.MeshStandardMaterial = {
  vertexUniforms: [],
  fragmentUniforms: [{
    name: 'emissive',
    format: 'color'
  }, {
    name: 'roughness',
    format: 'float'
  }, {
    name: 'metalness',
    format: 'float'
  }],
  textures: [{
    name: 'map',
    format: 'texture',
    define: 'USE_MAP'
  }, {
    name: 'emissiveMap',
    format: 'texture',
    define: 'USE_EMISSIVEMAP'
  }, {
    name: 'normalMap',
    format: 'texture',
    define: 'USE_NORMALMAP'
  }, {
    name: 'roughnessMap',
    format: 'texture',
    define: 'USE_ROUGHNESSMAP'
  }, {
    name: 'metalnessMap',
    format: 'texture',
    define: 'USE_METALNESSMAP'
  }, {
    name: 'aoMap',
    format: 'texture',
    define: 'USE_AOMAP'
  }],
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

  layout(location = 0) out vec3 fragPosition;
  layout(location = 1) out vec3 fragNormal;
  layout(location = 2) out vec2 fragUv;

  void main() {
    gl_Position = uniforms.projectionMatrix * uniforms.viewMatrix * uniforms.modelMatrix * vec4(position, 1.0);
    fragPosition = position;
    fragNormal = normalize((uniforms.modelMatrix * vec4(normal, 0.0)).xyz);
    fragUv = uv;
  }`,
  fragmentShaderCode: `
  layout(set=0, binding=1) uniform Uniforms {
    vec3 diffuse;
    float opacity;
    vec3 emissive;
    float roughness;
    float metalness;
  } uniforms;

  // @TODO: Cut off the binding nums of the unused maps

  #ifdef USE_MAP
    layout(set=0, binding=MAP_BINDING) uniform sampler mapSampler;
    layout(set=0, binding=MAP_SAMPLER_BINDING) uniform texture2D map;
  #endif

  #ifdef USE_EMISSIVEMAP
    layout(set=0, binding=EMISSIVEMAP_BINDING) uniform sampler emissiveMapSampler;
    layout(set=0, binding=EMISSIVEMAP_SAMPLER_BINDING) uniform texture2D emissiveMap;
  #endif

  #ifdef USE_NORMALMAP
    layout(set=0, binding=NORMALMAP_BINDING) uniform sampler normalMapSampler;
    layout(set=0, binding=NORMALMAP_SAMPLER_BINDING) uniform texture2D normalMap;
  #endif

  #ifdef USE_ROUGHNESSMAP
    layout(set=0, binding=ROUGHNESSMAP_BINDING) uniform sampler roughnessMapSampler;
    layout(set=0, binding=ROUGHNESSMAP_SAMPLER_BINDING) uniform texture2D roughnessMap;
  #endif

  #ifdef USE_METALNESSMAP
    layout(set=0, binding=METALNESSMAP_BINDING) uniform sampler metalnessMapSampler;
    layout(set=0, binding=METALNESSMAP_SAMPLER_BINDING) uniform texture2D metalnessMap;
  #endif

  #ifdef USE_AOMAP
    layout(set=0, binding=AOMAP_BINDING) uniform sampler aoMapSampler;
    layout(set=0, binding=AOMAP_SAMPLER_BINDING) uniform texture2D aoMap;
  #endif

  layout(location = 0) in vec3 fragPosition;
  layout(location = 1) in vec3 fragNormal;
  layout(location = 2) in vec2 fragUv;
  layout(location = 0) out vec4 outColor;

  const float M_PI = 3.141592653589793;

  struct PBRInfo {
    float NdotL;
    float NdotV;
    float NdotH;
    float LdotH;
    float VdotH;
    float roughness;
    float metallic;
    vec3 reflectance0;
    vec3 reflectance90;
    float alphaRoughness;
    vec3 diffuseColor;
    vec3 specularColor;
  };

  vec3 getNormal() {
    #ifdef USE_NORMALMAP
      vec3 pos_dx = dFdx(fragPosition);
      vec3 pos_dy = dFdy(fragPosition);
      vec3 tex_dx = dFdx(vec3(fragUv, 0.0));
      vec3 tex_dy = dFdy(vec3(fragUv, 0.0));
      vec3 t = (tex_dy.t * pos_dx - tex_dx.t * pos_dy) / (tex_dx.s * tex_dy.t - tex_dy.s * tex_dx.t);

      vec3 ng = normalize(fragNormal);
      t = normalize(t - ng * dot(ng, t));
      vec3 b = normalize(cross(ng, t));
      mat3 tbn = mat3(t, b, ng);

      vec3 n = texture(sampler2D(normalMap, normalMapSampler), fragUv).rgb;

      float normalScale = 1.0;
      n = normalize(tbn * ((2.0 * n - 1.0) * vec3(normalScale, normalScale, 1.0)));

      return n;
    #else
      // @TODO: Check if this is correct
      return normalize(fragNormal);
    #endif
  }

  vec3 specularReflection(PBRInfo pbrInputs) {
    return pbrInputs.reflectance0
      + (pbrInputs.reflectance90 - pbrInputs.reflectance0)
      * pow(clamp(1.0 - pbrInputs.VdotH, 0.0, 1.0), 5.0);
  }

  float geometricOcclusion(PBRInfo pbrInputs) {
    float NdotL = pbrInputs.NdotL;
    float NdotV = pbrInputs.NdotV;
    float r = pbrInputs.alphaRoughness;

    float attenuationL = 2.0 * NdotL / (NdotL + sqrt(r * r + (1.0 - r * r) * (NdotL * NdotL)));
    float attenuationV = 2.0 * NdotV / (NdotV + sqrt(r * r + (1.0 - r * r) * (NdotV * NdotV)));
    return attenuationL * attenuationV;
  }

  float microfacetDistribution(PBRInfo pbrInputs) {
    float roughnessSq = pbrInputs.alphaRoughness * pbrInputs.alphaRoughness;
    float f = (pbrInputs.NdotH * roughnessSq - pbrInputs.NdotH) * pbrInputs.NdotH + 1.0;
    return roughnessSq / (M_PI * f * f);
  }

  vec3 diffuse(PBRInfo pbrInputs) {
    return pbrInputs.diffuseColor / M_PI;
  }

  vec4 sRGBToLinear(in vec4 value) {
    return vec4(mix(pow(value.rgb * 0.9478672986 + vec3(0.0521327014), vec3(2.4)), value.rgb * 0.0773993808, vec3(lessThanEqual(value.rgb, vec3(0.04045)))), value.a);
  }

  vec4 LinearTosRGB(in vec4 value) {
    return vec4(mix(pow(value.rgb, vec3(0.41666)) * 1.055 - vec3(0.055), value.rgb * 12.92, vec3(lessThanEqual(value.rgb, vec3(0.0031308)))), value.a);
  }

  void main() {
    vec4 baseColor = vec4(uniforms.diffuse, uniforms.opacity);
    #ifdef USE_MAP
      vec4 texelColor = texture(sampler2D(map, mapSampler), fragUv);
      texelColor = sRGBToLinear(texelColor);
      baseColor *= texelColor;
    #endif
    float roughnessFactor = uniforms.roughness;
    #ifdef USE_ROUGHNESSMAP
      vec4 texelRoughness = texture(sampler2D(roughnessMap, roughnessMapSampler), fragUv);
      roughnessFactor *= texelRoughness.g;
    #endif
    roughnessFactor = clamp(roughnessFactor, 0.04, 1.0);
    float metalnessFactor = uniforms.metalness;
    #ifdef USE_METALNESSMAP
      vec4 texelMetalness = texture(sampler2D(metalnessMap, metalnessMapSampler), fragUv);
      metalnessFactor *= texelMetalness.b;
    #endif
    metalnessFactor = clamp(metalnessFactor, 0.0, 1.0);

    float alphaRoughness = roughnessFactor * roughnessFactor;

    vec3 f0 = vec3(0.04);
    vec3 diffuseColor = baseColor.rgb * (vec3(1.0) - f0);
    diffuseColor *= 1.0 - metalnessFactor;
    vec3 specularColor = mix(f0, baseColor.rgb, metalnessFactor);

    float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);
    float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
    vec3 specularEnvironmentR0 = specularColor.rgb;
    vec3 specularEnvironmentR90 = vec3(1.0) * reflectance90;

    vec3 camera = vec3(0.0, 0.0, 2.0); // @TODO: Should be from Camera object
    vec3 lightDirection = vec3(1.0, 2.0, 2.0); // @TODO: Should be from Light nodes
    vec3 n = getNormal();
    vec3 v = normalize(camera - fragPosition);
    vec3 l = normalize(lightDirection);
    vec3 h = normalize(l + v);
    vec3 reflection = -normalize(reflect(v, n));

    float NdotL = clamp(dot(n, l), 0.001, 1.0);
    float NdotV = clamp(abs(dot(n, v)), 0.001, 1.0);
    float NdotH = clamp(dot(n, h), 0.0, 1.0);
    float LdotH = clamp(dot(l, h), 0.0, 1.0);
    float VdotH = clamp(dot(v, h), 0.0, 1.0);

    PBRInfo pbrInputs = PBRInfo(
      NdotL,
      NdotV,
      NdotH,
      LdotH,
      VdotH,
      roughnessFactor,
      metalnessFactor,
      specularEnvironmentR0,
      specularEnvironmentR90,
      alphaRoughness,
      diffuseColor,
      specularColor
    );

    vec3 F = specularReflection(pbrInputs);
    float G = geometricOcclusion(pbrInputs);
    float D = microfacetDistribution(pbrInputs);

    vec3 diffuseContrib = (1.0 - F) * diffuse(pbrInputs);
    vec3 specContrib = F * G * D / (4.0 * NdotL * NdotV);

    // @TODO: Should be configurable
    float lightIntensity = 2.0;
    vec3 lightColor = vec3(1.0) * lightIntensity;
    vec3 color = NdotL * lightColor * (diffuseContrib + specContrib);

    #ifdef USE_AOMAP
      float occlusionStrength = 1.0;
      float ao = texture(sampler2D(aoMap, aoMapSampler), fragUv).r;
      color = mix(color, color * ao, occlusionStrength);
    #endif

    vec3 totalEmissiveRadiance = uniforms.emissive;
    #ifdef USE_EMISSIVEMAP
      vec4 emissiveColor = texture(sampler2D(emissiveMap, emissiveMapSampler), fragUv);
      emissiveColor.rgb = sRGBToLinear(emissiveColor).rgb;
      totalEmissiveRadiance *= emissiveColor.rgb;
    #endif
    color += totalEmissiveRadiance;

    outColor = vec4(color, baseColor.a);
    outColor = LinearTosRGB(outColor);
  }`
};

ShaderLibs.MeshBasicMaterial = {
  vertexUniforms: [],
  fragmentUniforms: [],
  textures: [{
    name: 'map',
    format: 'texture',
    define: 'USE_MAP'
  }],
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
    vec3 diffuse;
    float opacity;
  } uniforms;

  #ifdef USE_MAP
    layout(set=0, binding=MAP_BINDING) uniform sampler mapSampler;
    layout(set=0, binding=MAP_SAMPLER_BINDING) uniform texture2D map;
  #endif

  layout(location = 0) in vec3 fragNormal;
  layout(location = 1) in vec2 fragUv;
  layout(location = 0) out vec4 outColor;

  void main() {
    outColor = vec4(uniforms.diffuse, uniforms.opacity);
    #ifdef USE_MAP
      outColor *= texture(sampler2D(map, mapSampler), fragUv);
    #endif
  }`
};

ShaderLibs.MeshNormalMaterial = {
  vertexUniforms: [],
  fragmentUniforms: [],
  textures: [],
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
    vec3 diffuse;
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
    if (!options.device || !options.glslang) {
      throw new Error('WebGPURenderer: the constructor must take device and glslang parameters.');
    }

    this.domElement = document.createElement('canvas');
    this.context = this.domElement.getContext('gpupresent');

    this._device = options.device;
    this._glslang = options.glslang;

    this._projScreenMatrix = new Matrix4();
    this._width = 640;
    this._height = 480;
    this._pixelRatio = 1.0;

    this._format = 'bgra8unorm';
    this._sampleCount = 4;
    this._passEncoder = null;
    this._commandEncoder = null;
    this._colorAttachment = null;
    this._depthStencilAttachment = null;

    this._cache = {
      currentProgram: null
    };

    this._verticesManager = new WebGPUVerticesManager();
    this._indicesManager = new WebGPUIndicesManager();
    this._uniformsManager = new WebGPUUniformsManager();
    this._textureManager = new WebGPUTextureManager();
    this._programManager = new WebGPUProgramManager(this._glslang, this._textureManager);
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
  constructor(glslang, textureManager) {
    this.glslang = glslang;
    this.textureManager = textureManager;
    this.map = new Map(); // material -> program
    this.map2 = new Map(); // material type key -> program
  }

  get(material, device) {
    if (!this.map.has(material)) {
      // @TODO: Create key more properly
      const key = material.type + ':' + [
        !!material.map,
        !!material.emissiveMap,
        !!material.normalMap,
        !!material.roughnessMap,
        !!material.metalnessMap,
        !!material.aoMap
      ].join(':');
      if (!this.map2.has(key)) {
        this.map2.set(key, new WebGPUProgram(
          this.glslang,
          material,
          this.textureManager,
          device
        ));
      }
      this.map.set(material, this.map2.get(key));
    }
    return this.map.get(material);
  }
}

class WebGPUProgram {
  constructor(glslang, material, textureManager, device) {
    this.textureManager = textureManager;
    const shader = ShaderLibs[material.type];

    if (!shader) {
      throw new Error('This type of material is not supported yet. ' +  key);
    }

    const entries = [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      type: 'uniform-buffer'
    }, {
      binding: 1,
      visibility: GPUShaderStage.FRAGMENT,
      type: 'uniform-buffer'
    }];

    let binding = 2;
    const defines = [];
    for (const definition of shader.textures) {
      switch (definition.name) {
        case 'map':
          if (!material.map) {
            continue;
          }
          break;
        case 'emissiveMap':
          if (!material.emissiveMap) {
            continue;
          }
          break;
        case 'normalMap':
          if (!material.normalMap) {
            continue;
          }
          break;
        case 'roughnessMap':
          if (!material.roughnessMap) {
            continue;
          }
          break;
        case 'metalnessMap':
          if (!material.metalnessMap) {
            continue;
          }
          break;
        case 'aoMap':
          if (!material.aoMap) {
            continue;
          }
          break;
        default:
          console.error('Unknown texture ' + definition.name);
          continue;
      }
      entries.push({
        binding: binding,
        visibility: GPUShaderStage.FRAGMENT,
        type: 'sampler'
      });
      entries.push({
        binding: binding + 1,
        visibility: GPUShaderStage.FRAGMENT,
        type: 'sampled-texture'
      });
      if (definition.define) {
        defines.push('#define ' + definition.define);
        defines.push('#define ' + definition.name.toUpperCase() + '_BINDING ' + binding);
        defines.push('#define ' + definition.name.toUpperCase() + '_SAMPLER_BINDING ' + (binding + 1));
      }
      binding += 2;
    }

    const vertexShaderCode = '#version 450\n' +
      defines.join('\n') + '\n' +
      shader.vertexShaderCode;

    const fragmentShaderCode = '#version 450\n' +
      defines.join('\n') + '\n' +
      shader.fragmentShaderCode;

    this.uniformGroupLayout = device.createBindGroupLayout({
      entries: entries,
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
      vertexState: this._createVertexState(),
      colorStates: [{
        format: 'bgra8unorm',
        colorBlend: {
          srcFactor: 'src-alpha',
          dstFactor: 'one-minus-src-alpha',
          operation: 'add'
        }
      }],
      sampleCount: 4 // @TODO: Should be configurable
    });
  }

  _createVertexState() {
    let offset = 0;
    const attributes = [];
    for (let i = 0; i < defaultAttributeDefinitions.length; i++) {
      const definition = defaultAttributeDefinitions[i];
      attributes.push({
        shaderLocation: i,
        offset: offset,
        format: definition.format
      });
      offset += offsetTable[definition.format];
    }

    return {
      indexFormat: 'uint16', // @TODO: Should be configurable
      vertexBuffers: [{
        arrayStride: offset,
        attributes: attributes
      }]
    };
  }

  createVertices(geometry, device) {
    let itemSize = 0;
    for (const definition of defaultAttributeDefinitions) {
      itemSize += offsetTable[definition.format];
    }
    const position = geometry.getAttribute('position');
    return new WebGPUVertices(new Float32Array(itemSize * position.count), device);
  }

  createUniforms(device, material) {
    const buffers = this._createUniformBuffers(material, device);
    const textures = this._createUniformTextures(material, device);
    const bindGroup = this._createUniformBindGroup(buffers, textures, device);
    return new WebGPUUniforms(buffers, bindGroup);
  }

  _createUniformBuffers(material, device) {
    const shader = ShaderLibs[material.type];

    let vertexUniformsSize = 0;
    for (const definition of defaultVertexUniformsDefinitions) {
      vertexUniformsSize += offsetTable[definition.format];
    }
    for (const definition of shader.vertexUniforms) {
      vertexUniformsSize += offsetTable[definition.format];
    }

    let fragmentUniformsSize = 0;
    for (const definition of defaultFragmentUniformsDefinitions) {
      fragmentUniformsSize += offsetTable[definition.format];
    }
    for (const definition of shader.fragmentUniforms) {
      fragmentUniformsSize += offsetTable[definition.format];
    }

    const buffers = [];
    buffers.push(new WebGPUUniformBuffer(vertexUniformsSize, device)); // vertex
    buffers.push(new WebGPUUniformBuffer(fragmentUniformsSize, device)); // fragment
    return buffers;
  }

  _createUniformTextures(material, device) {
    const textures = [];
    for (const definition of ShaderLibs[material.type].textures) {
      switch (definition.name) {
        case 'map':
          if (material.map) {
            textures.push(this.textureManager.get(material.map, device));
          }
          break;
        case 'emissiveMap':
          if (material.emissiveMap) {
            textures.push(this.textureManager.get(material.emissiveMap, device));
          }
          break;
        case 'normalMap':
          if (material.normalMap) {
            textures.push(this.textureManager.get(material.normalMap, device));
          }
          break;
        case 'roughnessMap':
          if (material.roughnessMap) {
            textures.push(this.textureManager.get(material.roughnessMap, device));
          }
          break;
        case 'metalnessMap':
          if (material.metalnessMap) {
            textures.push(this.textureManager.get(material.metalnessMap, device));
          }
          break;
        case 'aoMap':
          if (material.aoMap) {
            textures.push(this.textureManager.get(material.aoMap, device));
          }
          break;
        default:
          console.error('Unknown texture ' + definition.name);
          continue;
      }
    }
    return textures;
  }

  _createUniformBindGroup(buffers, textures, device) {
    const entries = [];
    for (let i = 0; i < buffers.length; i++) {
      const buffer = buffers[i];
      entries.push({
        binding: i,
        resource: {
          buffer: buffer.buffer,
          size: buffer.byteLength
        }
      });
    }
    for (let i = 0; i < textures.length; i++) {
      entries.push({
        binding: entries.length,
        resource: textures[i].sampler.sampler
      });
      entries.push({
        binding: entries.length,
        resource: textures[i].texture.createView()
      });
    }
    return device.createBindGroup({
      layout: this.uniformGroupLayout,
      entries: entries
    });
  }
}

class WebGPUUniformsManager {
  constructor() {
    this.map = new Map();
  }

  get(object, camera, program, device) {
    if (!this.map.has(object)) {
      this.map.set(object, program.createUniforms(device, object.material));
    }
    const uniforms = this.map.get(object);
    uniforms.update(object, camera, device);
    return uniforms;
  }
}

class WebGPUUniforms {
  constructor(buffers, bindGroup) {
    this.buffers = buffers;
    this.bindGroup = bindGroup;
  }

  update(object, camera, device) {
    object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
    object.normalMatrix.getNormalMatrix(object.modelViewMatrix);

    const shader = ShaderLibs[object.material.type];

    // for vertex shader
    let offset = 0;
    const vertexUniformsBuffer = this.buffers[0];
    for (const definition of defaultVertexUniformsDefinitions) {
      let value;
      switch (definition.name) {
        case 'modelMatrix':
          value = object.matrixWorld;
          break;
        case 'viewMatrix':
          value = camera.matrixWorldInverse;
          break;
        case 'projectionMatrix':
          value = camera.projectionMatrix;
          break;
        case 'normalMatrix':
          value = object.normalMatrix;
          break;
        default:
          console.error('Unknown uniform ' + definition.name);
          continue;
      }
      if (value) {
        this._updateData(vertexUniformsBuffer, definition.format, offset, value);
      }
      offset += offsetTable[definition.format];
    }
    vertexUniformsBuffer.upload(device);

    // for fragment shader
    offset = 0;
    const fragmentUniformsBuffer = this.buffers[1];
    for (const definition of defaultFragmentUniformsDefinitions) {
      let value;
      switch (definition.name) {
        case 'diffuse':
          value = object.material.color;
          break;
        case 'opacity':
          value = object.material.opacity;
          break;
        default:
          console.error('Unknown uniform ' + definition.name);
          continue;
      }
      if (value !== undefined && value !== null) {
        this._updateData(fragmentUniformsBuffer, definition.format, offset, value);
      }
      offset += offsetTable[definition.format];
    }

    for (const definition of shader.fragmentUniforms) {
      let value;
      switch (definition.name) {
        case 'emissive':
          value = object.material.emissive;
          break;
        case 'roughness':
          value = object.material.roughness;
          break;
        case 'metalness':
          value = object.material.metalness;
          break;
        default:
          console.error('Unknown uniform ' + definition.name);
          continue;
      }
      if (value !== undefined && value !== null) {
        this._updateData(fragmentUniformsBuffer, definition.format, offset, value);
      }
      offset += offsetTable[definition.format];
    }
    fragmentUniformsBuffer.upload(device);
  }

  _updateData(buffer, format, offset, value) {
    switch (format) {
      case 'float':
        buffer.updateFloat(offset, value);
        break;
      case 'float3':
        buffer.updateVector3(offset, value);
        break;
      case 'color':
        buffer.updateColor(offset, value);
        break;
      case 'matrix3':
        buffer.updateMatrix3(offset, value);
        break;
      case 'matrix4':
        buffer.updateMatrix4(offset, value);
        break;
      default:
        console.error('Unknown format ' + format);
        break;
    }
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

  upload(device) {
    device.defaultQueue.writeBuffer(this.buffer, 0, this.float32Array.buffer);
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
    // @TODO: Should be configurable
    this.sampler = device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear'
    });
  }
}

class WebGPUTextureManager {
  constructor() {
    this.map = new Map();
  }

  get(texture, device) {
    if (!this.map.has(texture)) {
      const image = texture.image;
      const webgpuTexture = new WebGPUTexture(image.width, image.height, device);
      webgpuTexture.upload(image, device);
      this.map.set(texture, webgpuTexture);
    }
    return this.map.get(texture);
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
    device.defaultQueue.writeBuffer(this.buffer, 0, imageData.data.buffer);

    const encoder = device.createCommandEncoder({});
    encoder.copyBufferToTexture({
      buffer: this.buffer,
      bytesPerRow: this.width * 4,
      rowsPerImage: 0
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
      vertices.upload(device);
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

    // @TODO: Remove this temporal workaround
    if (!geometry.getAttribute('uv')) {
      geometry.setAttribute('uv', new BufferAttribute(new Float32Array(position.count * 2), 2));
    }
    const uv = geometry.getAttribute('uv');

    let dataIndex = 0;
    const indices = {};
    for (const definition of defaultAttributeDefinitions) {
      indices[definition.name] = 0;
    }
    for (let i = 0; i < position.count; i++) {
      for (const definition of defaultAttributeDefinitions) {
        const attribute = geometry.getAttribute(definition.name);
        for (let j = 0; j < attribute.itemSize; j++) {
          this.array[dataIndex++] = attribute.array[indices[definition.name]++];
        }
      }
    }
  }

  upload(device) {
    device.defaultQueue.writeBuffer(this.buffer, 0, this.array.buffer);
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
      indices.upload(device);
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

  upload(device) {
    device.defaultQueue.writeBuffer(this.buffer, 0, this.array.buffer);
  }
}
