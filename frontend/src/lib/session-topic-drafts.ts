export type SessionTopicDrafts = Record<string, string>;

export function getSessionTopicDraft(
  drafts: SessionTopicDrafts,
  sessionId: string | null
): string {
  if (!sessionId) {
    return "";
  }
  return drafts[sessionId] ?? "";
}

export function setSessionTopicDraft(
  drafts: SessionTopicDrafts,
  sessionId: string | null,
  topic: string
): SessionTopicDrafts {
  if (!sessionId) {
    return drafts;
  }

  if (topic === "") {
    const nextDrafts = { ...drafts };
    delete nextDrafts[sessionId];
    return nextDrafts;
  }

  return {
    ...drafts,
    [sessionId]: topic,
  };
}

export function removeSessionTopicDraft(
  drafts: SessionTopicDrafts,
  sessionId: string | null
): SessionTopicDrafts {
  if (!sessionId || !(sessionId in drafts)) {
    return drafts;
  }

  const nextDrafts = { ...drafts };
  delete nextDrafts[sessionId];
  return nextDrafts;
}

export function normalizeSessionTopicDrafts(
  drafts: unknown
): SessionTopicDrafts {
  if (!drafts || typeof drafts !== "object") {
    return {};
  }

  return Object.fromEntries(
    Object.entries(drafts).filter(
      ([sessionId, topic]) =>
        typeof sessionId === "string" &&
        sessionId.length > 0 &&
        typeof topic === "string" &&
        topic.length > 0
    )
  );
}

export function migrateLegacyTopicDraftState(
  persistedState: unknown
): Record<string, unknown> {
  if (!persistedState || typeof persistedState !== "object") {
    return { sessionTopicDrafts: {} };
  }

  const {
    topic,
    currentSessionId,
    sessionTopicDrafts,
    ...rest
  } = persistedState as {
    topic?: unknown;
    currentSessionId?: unknown;
    sessionTopicDrafts?: unknown;
    [key: string]: unknown;
  };

  const normalizedDrafts = normalizeSessionTopicDrafts(sessionTopicDrafts);

  if (
    typeof topic === "string" &&
    topic.length > 0 &&
    typeof currentSessionId === "string" &&
    currentSessionId.length > 0 &&
    !(currentSessionId in normalizedDrafts)
  ) {
    normalizedDrafts[currentSessionId] = topic;
  }

  return {
    ...rest,
    currentSessionId:
      typeof currentSessionId === "string" || currentSessionId === null
        ? currentSessionId
        : null,
    sessionTopicDrafts: normalizedDrafts,
  };
}
