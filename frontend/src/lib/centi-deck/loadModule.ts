import gsap from "gsap";
import { Flip } from "gsap/Flip";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { SplitText } from "gsap/SplitText";
import { DrawSVGPlugin } from "gsap/DrawSVGPlugin";
import { MorphSVGPlugin } from "gsap/MorphSVGPlugin";

import type {
  CentiDeckSlideDescriptor,
  CentiDeckSlideModule,
  LoadedCentiDeckSlide,
} from "./types";

let pluginsRegistered = false;

function ensurePluginsRegistered(): void {
  if (pluginsRegistered) return;
  gsap.registerPlugin(Flip, ScrollTrigger, SplitText, DrawSVGPlugin, MorphSVGPlugin);
  pluginsRegistered = true;
}

function prefixWithStrictPreamble(source: string): string {
  const trimmed = source.trimStart();
  if (trimmed.startsWith('"use strict"') || trimmed.startsWith("'use strict'")) {
    return source;
  }
  return `"use strict";\n${source}`;
}

/**
 * Dynamically import an agent-authored ES module via a blob URL.
 *
 * Same-origin execution: the module has full access to the page. The backend's
 * `normalize_centi_deck_submission` pre-validated the source against a regex
 * whitelist and prepended `"use strict"`. This loader double-checks the strict
 * preamble before handing the blob to `import()`.
 */
export async function loadCentiDeckModule(
  descriptor: CentiDeckSlideDescriptor
): Promise<LoadedCentiDeckSlide> {
  ensurePluginsRegistered();

  const sanitizedSource = prefixWithStrictPreamble(descriptor.moduleSource);
  const blob = new Blob([sanitizedSource], { type: "application/javascript" });
  const blobUrl = URL.createObjectURL(blob);

  try {
    const imported = (await import(/* @vite-ignore */ /* webpackIgnore: true */ blobUrl)) as {
      default?: CentiDeckSlideModule;
    };
    const slideModule = imported?.default;
    if (!slideModule || typeof slideModule.render !== "function") {
      throw new Error(
        `Centi-deck module for slide ${descriptor.slideId} must export default { render() }`
      );
    }
    return { descriptor, module: slideModule };
  } finally {
    // Revoke immediately — the module is already instantiated.
    URL.revokeObjectURL(blobUrl);
  }
}

export async function loadAllCentiDeckModules(
  descriptors: CentiDeckSlideDescriptor[]
): Promise<LoadedCentiDeckSlide[]> {
  return Promise.all(descriptors.map(loadCentiDeckModule));
}
