"use client";

import { useCallback, useRef, useState } from "react";

/**
 * Copying, with the failure made visible.
 *
 * Every copy button here used to be `try { await navigator.clipboard… }
 * catch {}`, which has two failure modes and no symptom for either. The
 * promise can reject - no permission, not a secure context - and it can also
 * simply never settle, which was observed: the button stayed on "copy"
 * forever and the reader had no way to tell a slow copy from a dead one.
 *
 * So: race the write against a deadline, fall back to the old
 * execCommand path, and return whether it worked. A button that cannot copy
 * should say so, because the reader's next move - select the text by hand -
 * depends on knowing.
 */

const DEADLINE_MS = 1500;

function legacyCopy(text: string): boolean {
  if (typeof document === "undefined") return false;
  const field = document.createElement("textarea");
  field.value = text;
  // Off-screen rather than hidden: a display:none field cannot be selected.
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.top = "-1000px";
  field.style.opacity = "0";
  document.body.appendChild(field);
  try {
    field.select();
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    field.remove();
  }
}

export async function copyText(text: string): Promise<boolean> {
  if (typeof navigator !== "undefined" && navigator.clipboard) {
    try {
      const wrote = await Promise.race([
        navigator.clipboard.writeText(text).then(() => true),
        new Promise<false>((resolve) => setTimeout(() => resolve(false), DEADLINE_MS)),
      ]);
      if (wrote) return true;
    } catch {
      // fall through to the legacy path
    }
  }
  return legacyCopy(text);
}

export type CopyState = "idle" | "done" | "failed";

/** `[state, copy]`, where state returns to idle on its own. */
export function useCopy(hold = 1400): [CopyState, (text: string) => void] {
  const [state, setState] = useState<CopyState>("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const copy = useCallback(
    (text: string) => {
      if (timer.current) clearTimeout(timer.current);
      void copyText(text).then((ok) => {
        setState(ok ? "done" : "failed");
        // A failure is worth leaving up longer than a success: it is asking
        // the reader to do something instead of telling them it is done.
        timer.current = setTimeout(() => setState("idle"), ok ? hold : hold * 2);
      });
    },
    [hold],
  );

  return [state, copy];
}
