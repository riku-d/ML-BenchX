/**
 * Which of the workspace's two layouts the reader is in.
 *
 * The product leads with code: at 1440px the Monaco editor is the largest thing on the
 * screen, so the first thing a student meets is Python. The fix is **not** to demote the
 * editor — `docs/PRODUCT.md` is explicit that it is central, and that hand-edited code
 * becoming authoritative is a core product behaviour. So there are two layouts and the
 * reader picks one:
 *
 * * **Explain** gives the centre column to the pipeline, where every step explains
 *   itself. The editor keeps its own column and stays visible.
 * * **Code** is the existing layout, unchanged, with the editor in the centre.
 *
 * Both contain every panel. Neither hides a feature; only the widths differ.
 *
 * The mode lives in the URL hash, because nothing else in this application's UI state
 * survived a reload. It is also remembered, so a reader who chose one does not have to
 * choose again on the next dataset. A reader who has hand-edited the code is treated as
 * working in code and defaults to that layout — but the stored choice outranks the
 * guess, because a stated preference beats an inference about one.
 */

export type ViewMode = "explain" | "code";

const STORAGE_KEY = "mlbenchx.workspace.view";

export function parseViewMode(raw: string | null | undefined): ViewMode | null {
  return raw === "explain" || raw === "code" ? raw : null;
}

/** The mode this reader last chose, or null if they never have. */
export function readStoredViewMode(): ViewMode | null {
  try {
    return parseViewMode(window.localStorage.getItem(STORAGE_KEY));
  } catch {
    // Private browsing and blocked storage both throw here. Falling back to the
    // default is correct; failing to render the workspace would not be.
    return null;
  }
}

export function storeViewMode(mode: ViewMode): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    // Nothing to do: the mode still works for this visit, it just is not remembered.
  }
}

/**
 * Which layout to show, in order of authority: the URL, then the stored choice, then
 * whether this workspace's code has been taken over by hand.
 */
export function resolveViewMode(
  fromUrl: ViewMode | null,
  codeIsAuthoritative: boolean,
): ViewMode {
  return fromUrl ?? readStoredViewMode() ?? (codeIsAuthoritative ? "code" : "explain");
}
