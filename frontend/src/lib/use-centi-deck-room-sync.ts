import { useEffect, useRef } from "react";

export interface CentiDeckRoomMessage {
  type: "centi-deck:slide";
  slideIndex: number;
  origin: string;
}

export interface UseCentiDeckRoomSyncOptions {
  sessionId: string | null | undefined;
  slideIndex: number;
  onRemoteSlideIndex?: (slideIndex: number) => void;
  /** A short origin label so a room can distinguish its own echoes. */
  originLabel?: string;
}

/**
 * Cross-window/tab sync for centi-deck deck navigation.
 *
 * Uses BroadcastChannel keyed by sessionId. When the local slide index
 * changes, publishes. Incoming messages from other tabs/windows invoke
 * `onRemoteSlideIndex` unless the origin matches this hook's origin.
 *
 * Degrades gracefully when BroadcastChannel is unavailable.
 */
export function useCentiDeckRoomSync({
  sessionId,
  slideIndex,
  onRemoteSlideIndex,
  originLabel = "unknown",
}: UseCentiDeckRoomSyncOptions): void {
  const channelRef = useRef<BroadcastChannel | null>(null);
  const originRef = useRef<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    if (typeof BroadcastChannel === "undefined") return;
    if (!originRef.current) {
      const randomId =
        typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
          ? crypto.randomUUID()
          : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
      originRef.current = `${originLabel}:${randomId}`;
    }
    const channel = new BroadcastChannel(`centi-deck:${sessionId}`);
    channelRef.current = channel;
    const origin = originRef.current;

    const handler = (event: MessageEvent<CentiDeckRoomMessage>) => {
      const message = event.data;
      if (!message || message.type !== "centi-deck:slide") return;
      if (message.origin === origin) return;
      onRemoteSlideIndex?.(message.slideIndex);
    };
    channel.addEventListener("message", handler);

    return () => {
      channel.removeEventListener("message", handler);
      channel.close();
      if (channelRef.current === channel) {
        channelRef.current = null;
      }
    };
  }, [sessionId, onRemoteSlideIndex, originLabel]);

  useEffect(() => {
    const channel = channelRef.current;
    if (!channel) return;
    const message: CentiDeckRoomMessage = {
      type: "centi-deck:slide",
      slideIndex,
      origin: originRef.current ?? originLabel,
    };
    channel.postMessage(message);
  }, [slideIndex, originLabel]);
}
