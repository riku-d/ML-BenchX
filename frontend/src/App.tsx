/**
 * Top-level routing between the three screens.
 *
 * The route lives in the URL hash rather than in state alone, so a reload keeps you in
 * the workspace you were working in. That matters more here than in most apps: the
 * workspace holds edited code and a pipeline someone has spent time configuring, and
 * losing it to an accidental refresh would be the worst kind of data loss — the kind
 * the user caused and cannot undo.
 */

import { Suspense, lazy, useCallback, useEffect, useState } from "react";
import { Landing } from "./landing/Landing";
import { UploadScreen } from "./workspace/UploadScreen";
import { Workspace } from "./workspace/Workspace";
import { parseViewMode, storeViewMode, type ViewMode } from "./explain/viewMode";

/**
 * The plan screen is loaded on demand.
 *
 * It is the only screen that draws charts, and the charting library is a third of the
 * bundle. Everyone pays for it on first load if it is imported eagerly; this way only
 * the people who open the plan do, and they are already waiting for a network round
 * trip to fetch the plan itself.
 */
const PlanReview = lazy(() =>
  import("./workspace/PlanReview").then((module) => ({ default: module.PlanReview })),
);

type Route =
  | { screen: "landing" }
  | { screen: "upload" }
  /** `view` is null until the workspace has resolved one — see `resolveViewMode`. */
  | { screen: "workspace"; datasetId: string; view: ViewMode | null }
  | { screen: "plan"; datasetId: string };

function parseHash(): Route {
  const hash = window.location.hash.replace(/^#\/?/, "");
  if (hash.startsWith("plan/")) {
    const datasetId = hash.slice("plan/".length);
    if (datasetId) return { screen: "plan", datasetId };
  }
  if (hash.startsWith("workspace/")) {
    // `workspace/{id}?view=explain`. The id is read off the front rather than by
    // splitting, so an unrecognised parameter cannot swallow the dataset.
    const rest = hash.slice("workspace/".length);
    const [datasetId, query = ""] = rest.split("?");
    if (datasetId) {
      const view = parseViewMode(new URLSearchParams(query).get("view"));
      return { screen: "workspace", datasetId, view };
    }
  }
  if (hash === "upload") return { screen: "upload" };
  return { screen: "landing" };
}

function toHash(route: Route): string {
  if (route.screen === "plan") return `#/plan/${route.datasetId}`;
  if (route.screen === "workspace") {
    const query = route.view ? `?view=${route.view}` : "";
    return `#/workspace/${route.datasetId}${query}`;
  }
  if (route.screen === "upload") return "#/upload";
  return "#/";
}

export default function App() {
  const [route, setRoute] = useState<Route>(parseHash);

  useEffect(() => {
    const onHashChange = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const go = useCallback((next: Route) => {
    window.location.hash = toHash(next);
    setRoute(next);
  }, []);

  /**
   * Record the workspace's layout in the URL without adding a history entry.
   *
   * Switching layout is not navigation — someone who flips to the code twice and then
   * presses Back expects to leave the workspace, not to walk back through their own
   * layout changes. `replaceState` keeps the URL shareable and reload-proof without
   * filling the history stack. The mode is also stored, so the next dataset opens the
   * way this reader last chose.
   */
  const setWorkspaceView = useCallback((datasetId: string, view: ViewMode, remember: boolean) => {
    if (remember) storeViewMode(view);
    const hash = toHash({ screen: "workspace", datasetId, view });
    window.history.replaceState(null, "", hash);
    setRoute({ screen: "workspace", datasetId, view });
  }, []);

  if (route.screen === "plan") {
    return (
      <Suspense
        fallback={
          <div className="flex h-screen items-center justify-center text-muted">
            Loading the plan…
          </div>
        }
      >
        <PlanReview
          key={route.datasetId}
          datasetId={route.datasetId}
          onExit={() => go({ screen: "workspace", datasetId: route.datasetId, view: null })}
        />
      </Suspense>
    );
  }

  if (route.screen === "workspace") {
    return (
      <Workspace
        key={route.datasetId}
        datasetId={route.datasetId}
        view={route.view}
        onViewChange={(view, remember) => setWorkspaceView(route.datasetId, view, remember)}
        onExit={() => go({ screen: "upload" })}
        onReviewPlan={() => go({ screen: "plan", datasetId: route.datasetId })}
        onDeleted={() => go({ screen: "upload" })}
      />
    );
  }

  if (route.screen === "upload") {
    return (
      <UploadScreen
        onReady={(dataset) => go({ screen: "workspace", datasetId: dataset.id, view: null })}
        onBack={() => go({ screen: "landing" })}
      />
    );
  }

  return <Landing onStart={() => go({ screen: "upload" })} />;
}
