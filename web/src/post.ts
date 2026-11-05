import * as THREE from 'three';
import { N8AOPostPass } from 'n8ao';
import { BloomEffect, BrightnessContrastEffect, type Effect, EffectComposer, EffectPass, HueSaturationEffect, RenderPass,
  SMAAEffect, ToneMappingEffect, ToneMappingMode, VignetteEffect } from 'postprocessing';

// the Genoa fork keeps the ambient occlusion and the grade: no bloom (only the night lights needed it) and no
// tilt-shift blur; SMAA only off phones, whose 1.5 pixels per CSS pixel are smooth enough
export function makePost(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, smaa: boolean) {
  const tone = new ToneMappingEffect({ mode: ToneMappingMode.NEUTRAL });
  const composer = new EffectComposer(renderer, { frameBufferType: THREE.HalfFloatType, multisampling: 0 });
  composer.addPass(new RenderPass(scene, camera));
  const n8 = new N8AOPostPass(scene, camera, innerWidth, innerHeight);
  Object.assign(n8.configuration, { aoRadius: 6, distanceFalloff: 1.5, intensity: 2.2, halfRes: true, depthAwareUpsampling: true });
  composer.addPass(n8);
  const bloom = new BloomEffect({ intensity: 0.2, luminanceThreshold: 0.92, luminanceSmoothing: 0.15, mipmapBlur: true, radius: 0.55 });
  const grade: Effect[] = [tone, new HueSaturationEffect({ saturation: 0.12 }), new BrightnessContrastEffect({ brightness: 0, contrast: 0.06 }),
    new VignetteEffect({ offset: 0.3, darkness: 0.45 })];
  if (smaa) grade.push(new SMAAEffect());
  composer.addPass(new EffectPass(camera, ...grade));
  return { composer, n8, bloom, tone };
}
