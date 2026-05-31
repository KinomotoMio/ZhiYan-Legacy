import gsap from "gsap";

import { ensureCentiDeckRuntimeStyles } from "./styles";
import { applyCentiDeckTheme } from "./theme";
import { rewriteViewportUnits } from "./units";
import type {
  CentiDeckRuntimeMode,
  CentiDeckSlideContext,
  CentiDeckTheme,
  LoadedCentiDeckSlide,
} from "./types";

export interface CentiDeckRuntimeOptions {
  /** Called whenever the active slide changes (after enter() hook resolves). */
  onSlideChange?: (slideIndex: number) => void;
  /** UI mode controls transitions and nav visibility. Default: interactive. */
  mode?: CentiDeckRuntimeMode;
  /** Starting slide. Default: 0. */
  startSlide?: number;
}

interface SlideState {
  actionStates: Map<string, Record<string, unknown>>;
  cleanupFns: Array<() => void>;
}

const TRANSITION_DURATION_MS = 700;
const ENTER_DELAY_MS = 80;

/**
 * Port of centi-deck's runtime (`/Users/miochan/cttk/centi-deck/src/core/runtime.js`).
 *
 * Class-based per-instance variant (the original is a module singleton).
 * Each instance owns its DOM root, loaded slides, navigation state, and
 * lifecycle hook dispatch. Enables fix-preview to mount two decks side-by-side.
 */
export class CentiDeckRuntime {
  private readonly slideStates = new WeakMap<HTMLElement, SlideState>();

  private container: HTMLElement | null = null;
  private deckRoot: HTMLElement | null = null;
  private slides: LoadedCentiDeckSlide[] = [];
  private sections: HTMLElement[] = [];
  private currentIndex = 0;
  private isTransitioning = false;
  private mode: CentiDeckRuntimeMode = "interactive";
  private onSlideChange: ((slideIndex: number) => void) | null = null;
  private themeCleanup: (() => void) | null = null;
  private themeSnapshot: CentiDeckTheme | null = null;

  mount(
    container: HTMLElement,
    deck: {
      slides: LoadedCentiDeckSlide[];
      theme?: CentiDeckTheme | null;
    },
    options: CentiDeckRuntimeOptions = {}
  ): void {
    this.unmount();

    this.container = container;
    ensureCentiDeckRuntimeStyles();
    this.mode = options.mode ?? "interactive";
    this.onSlideChange = options.onSlideChange ?? null;
    this.slides = deck.slides;
    this.themeSnapshot = deck.theme ?? null;

    container.classList.add("centi-deck-root");
    container.dataset.centiMode = this.mode;

    this.themeCleanup = applyCentiDeckTheme(container, deck.theme);

    const deckRoot = document.createElement("div");
    deckRoot.className = "centi-deck-slides";
    container.appendChild(deckRoot);
    this.deckRoot = deckRoot;

    this.sections = [];
    for (let index = 0; index < this.slides.length; index += 1) {
      const { module: slideModule } = this.slides[index];
      const section = document.createElement("section");
      section.className = "centi-deck-slide";
      section.dataset.slideIndex = String(index);
      section.dataset.slideId = this.slides[index].descriptor.slideId;

      const content = typeof slideModule.render === "function" ? slideModule.render() : "";
      if (typeof content === "string") {
        section.innerHTML = rewriteViewportUnits(content);
      } else if (content instanceof HTMLElement) {
        section.appendChild(content);
      }

      this.slideStates.set(section, {
        actionStates: new Map(),
        cleanupFns: [],
      });

      deckRoot.appendChild(section);
      this.sections.push(section);
    }

    const startIndex = clampIndex(options.startSlide ?? 0, this.slides.length);
    this.currentIndex = startIndex;
    this.goTo(startIndex, { immediate: true });
  }

  unmount(): void {
    for (let index = 0; index < this.sections.length; index += 1) {
      this.cleanupSlide(index);
    }

    if (this.themeCleanup) {
      this.themeCleanup();
      this.themeCleanup = null;
    }

    if (this.container && this.deckRoot && this.deckRoot.parentElement === this.container) {
      this.container.removeChild(this.deckRoot);
    }

    if (this.container) {
      this.container.classList.remove("centi-deck-root");
      delete this.container.dataset.centiMode;
    }

    this.container = null;
    this.deckRoot = null;
    this.slides = [];
    this.sections = [];
    this.currentIndex = 0;
    this.isTransitioning = false;
    this.onSlideChange = null;
    this.themeSnapshot = null;
  }

  get slideCount(): number {
    return this.slides.length;
  }

  get activeIndex(): number {
    return this.currentIndex;
  }

  goTo(
    index: number,
    options: { immediate?: boolean } = {}
  ): void {
    if (!this.slides.length) return;
    const clamped = clampIndex(index, this.slides.length);
    if (!options.immediate && this.isTransitioning) return;

    const previousIndex = this.currentIndex;
    this.currentIndex = clamped;

    if (options.immediate || this.mode !== "interactive") {
      if (previousIndex !== clamped) {
        this.cleanupSlide(previousIndex);
        this.invokeLeave(previousIndex);
      }
      // Snap mode (initial mount, thumbnails, presenter initial): no transition.
      this.sections.forEach((section, sectionIndex) => {
        section.classList.remove("is-active", "is-exiting");
        this.resetSlide(section);
        if (sectionIndex === clamped) {
          section.classList.add("is-active");
          this.invokeEnter(clamped);
        }
      });
      this.emitSlideChange();
      return;
    }

    this.isTransitioning = true;
    const prevSection = this.sections[previousIndex];
    if (prevSection && previousIndex !== clamped) {
      this.cleanupSlide(previousIndex);
      prevSection.classList.remove("is-active");
      prevSection.classList.add("is-exiting");
      this.invokeLeave(previousIndex);
      window.setTimeout(() => {
        prevSection.classList.remove("is-exiting");
        this.resetSlide(prevSection);
      }, TRANSITION_DURATION_MS);
    }

    const nextSection = this.sections[clamped];
    if (nextSection) {
      window.setTimeout(() => {
        this.resetSlide(nextSection);
        nextSection.classList.add("is-active");
        this.invokeEnter(clamped);
      }, ENTER_DELAY_MS);
    }

    window.setTimeout(() => {
      this.isTransitioning = false;
      this.emitSlideChange();
    }, TRANSITION_DURATION_MS);
  }

  next(): void {
    this.goTo(this.currentIndex + 1);
  }

  prev(): void {
    this.goTo(this.currentIndex - 1);
  }

  private invokeEnter(index: number): void {
    const slide = this.slides[index];
    const section = this.sections[index];
    if (!slide || !section) return;
    try {
      slide.module.enter?.(section, this.makeContext(index));
    } catch (error) {
      console.warn(`[centi-deck] enter() threw for slide ${slide.descriptor.slideId}`, error);
    }
  }

  private invokeLeave(index: number): void {
    const slide = this.slides[index];
    const section = this.sections[index];
    if (!slide || !section) return;
    try {
      slide.module.leave?.(section, this.makeContext(index));
    } catch (error) {
      console.warn(`[centi-deck] leave() threw for slide ${slide.descriptor.slideId}`, error);
    }
  }

  private makeContext(index: number): CentiDeckSlideContext {
    const slide = this.slides[index];
    const section = this.sections[index];
    const state = section ? this.slideStates.get(section) : undefined;
    return {
      slideId: slide.descriptor.slideId,
      slideIndex: index,
      section,
      gsap,
      goTo: (targetIndex: number) => this.goTo(targetIndex),
      registerCleanup: (fn: () => void) => {
        if (state && typeof fn === "function") {
          state.cleanupFns.push(fn);
        }
      },
    };
  }

  private cleanupSlide(index: number): void {
    const section = this.sections[index];
    if (!section) return;
    const state = this.slideStates.get(section);
    if (!state) return;
    for (const fn of state.cleanupFns.splice(0)) {
      try {
        fn();
      } catch {
        // ignore cleanup errors
      }
    }
  }

  private resetSlide(section: HTMLElement): void {
    gsap.killTweensOf(section);
    section.querySelectorAll("*").forEach((el) => gsap.killTweensOf(el));
    gsap.set(section, { clearProps: "opacity,transform,filter" });
    section
      .querySelectorAll("*")
      .forEach((el) => gsap.set(el, { clearProps: "opacity,transform,filter" }));
  }

  private emitSlideChange(): void {
    this.onSlideChange?.(this.currentIndex);
  }
}

function clampIndex(index: number, length: number): number {
  if (!Number.isFinite(index) || length <= 0) return 0;
  return Math.max(0, Math.min(Math.trunc(index), length - 1));
}
