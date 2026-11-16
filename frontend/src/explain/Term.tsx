/**
 * A term the reader can look up where it stands.
 *
 * Wraps the shared `Tooltip` with a lookup into `vocabulary.ts`. A term that has no
 * entry renders as ordinary text rather than as a dotted-underlined promise of an
 * explanation that does not arrive — so adding `<Term>` around a word is safe before
 * its definition has been written, and the absence shows up as a missing affordance
 * rather than as an empty tooltip.
 */

import type { ReactNode } from "react";
import { Tooltip } from "../components/ui";
import { define } from "./vocabulary";

export function Term({
  name,
  children,
  placement = "top",
}: {
  /** The vocabulary key. Defaults to the visible text when children is a plain string. */
  name?: string;
  children: ReactNode;
  placement?: "top" | "bottom";
}) {
  const key = name ?? (typeof children === "string" ? children : "");
  const definition = define(key);

  if (!definition) return <>{children}</>;

  return (
    <Tooltip
      placement={placement}
      content={
        <>
          <p className="font-medium text-text">{definition.what}</p>
          {definition.why && <p className="mt-1.5 text-muted">{definition.why}</p>}
        </>
      }
    >
      {children}
    </Tooltip>
  );
}
