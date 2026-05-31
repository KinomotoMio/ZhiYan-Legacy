import type {
  LayoutType,
  SceneBackground,
  SceneBackgroundColorToken,
  SceneBackgroundEmphasis,
  SceneBackgroundPreset,
  Slide,
} from "@/types/slide";

export type SceneBackgroundEligibleLayout =
  | "intro-slide"
  | "section-header"
  | "outline-slide"
  | "quote-slide"
  | "thank-you";

interface SceneBackgroundRule {
  preset: SceneBackgroundPreset;
  emphasis: SceneBackgroundEmphasis;
  allowedEmphasis: readonly SceneBackgroundEmphasis[];
}

const SCENE_BACKGROUND_RULES: Record<SceneBackgroundEligibleLayout, SceneBackgroundRule> = {
  "intro-slide": {
    preset: "hero-glow",
    emphasis: "immersive",
    allowedEmphasis: ["balanced", "immersive"],
  },
  "section-header": {
    preset: "section-band",
    emphasis: "immersive",
    allowedEmphasis: ["balanced", "immersive"],
  },
  "outline-slide": {
    preset: "outline-grid",
    emphasis: "subtle",
    allowedEmphasis: ["subtle", "balanced"],
  },
  "quote-slide": {
    preset: "quote-focus",
    emphasis: "balanced",
    allowedEmphasis: ["balanced", "immersive"],
  },
  "thank-you": {
    preset: "closing-wash",
    emphasis: "immersive",
    allowedEmphasis: ["balanced", "immersive"],
  },
};

const COLOR_TOKENS = new Set<SceneBackgroundColorToken>([
  "primary",
  "secondary",
  "neutral",
]);

const EMPHASIS_ORDER: SceneBackgroundEmphasis[] = [
  "subtle",
  "balanced",
  "immersive",
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSceneBackgroundPreset(value: unknown): value is SceneBackgroundPreset {
  return (
    value === "hero-glow" ||
    value === "section-band" ||
    value === "outline-grid" ||
    value === "quote-focus" ||
    value === "closing-wash"
  );
}

function isSceneBackgroundEmphasis(value: unknown): value is SceneBackgroundEmphasis {
  return value === "subtle" || value === "balanced" || value === "immersive";
}

function normalizeSceneBackgroundEmphasis(
  emphasis: unknown,
  rule: SceneBackgroundRule
): SceneBackgroundEmphasis {
  if (!isSceneBackgroundEmphasis(emphasis)) {
    return rule.emphasis;
  }

  if (rule.allowedEmphasis.includes(emphasis)) {
    return emphasis;
  }

  const maxAllowedIndex = Math.max(
    ...rule.allowedEmphasis.map((value) => EMPHASIS_ORDER.indexOf(value))
  );
  return EMPHASIS_ORDER[maxAllowedIndex];
}

export function getSceneBackgroundRule(
  layoutId: string | undefined
): SceneBackgroundRule | null {
  if (!layoutId) return null;

  return (SCENE_BACKGROUND_RULES as Partial<Record<LayoutType | string, SceneBackgroundRule>>)[
    layoutId
  ] ?? null;
}

export function supportsSceneBackgroundLayout(
  layoutId: string | undefined
): layoutId is SceneBackgroundEligibleLayout {
  return getSceneBackgroundRule(layoutId) !== null;
}

export function normalizeSceneBackground(
  layoutId: string | undefined,
  background: unknown
): SceneBackground | null | undefined {
  const rule = getSceneBackgroundRule(layoutId);
  if (!rule) {
    return undefined;
  }

  if (background === undefined) {
    return undefined;
  }

  if (background === null) {
    return null;
  }

  if (!isRecord(background) || background.kind !== "scene") {
    return null;
  }

  return {
    kind: "scene",
    preset:
      isSceneBackgroundPreset(background.preset) && background.preset === rule.preset
        ? background.preset
        : rule.preset,
    emphasis: normalizeSceneBackgroundEmphasis(background.emphasis, rule),
    colorToken: COLOR_TOKENS.has(background.colorToken as SceneBackgroundColorToken)
      ? (background.colorToken as SceneBackgroundColorToken)
      : "primary",
  };
}

export function normalizeSlideSceneBackground<T extends Slide>(slide: T): T {
  const layoutId = slide.layoutId || slide.layoutType;
  const normalizedBackground = normalizeSceneBackground(layoutId, slide.background);

  if (normalizedBackground === undefined) {
    if (!Object.prototype.hasOwnProperty.call(slide, "background")) {
      return slide;
    }

    const rest = { ...slide };
    delete rest.background;
    return rest as T;
  }

  if (JSON.stringify(slide.background ?? null) === JSON.stringify(normalizedBackground)) {
    return slide;
  }

  return {
    ...slide,
    background: normalizedBackground,
  };
}
