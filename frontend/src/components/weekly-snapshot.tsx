"use client";

// Phase 7 — thin wrapper around the shared <SnapshotGallery>. The
// gallery body (drop zone, paste/drag handlers, optimistic upload
// placeholders, lightbox) lives in snapshot-gallery.tsx; this file
// binds the weekly-retro presentation (entityType, props rename) so
// existing callsites and tests continue to work without modification.
//
// The component's public API is unchanged from Phase 4.4:
//   <WeeklySnapshot
//     retroId={...}
//     portfolio={...}
//     onCountChange={...} />

import { SnapshotGallery } from "./snapshot-gallery";

export interface WeeklySnapshotProps {
  retroId: number | null;
  portfolio: string;
  onCountChange?: (count: number) => void;
}

export function WeeklySnapshot({
  retroId,
  portfolio,
  onCountChange,
}: WeeklySnapshotProps) {
  return (
    <SnapshotGallery
      entityType="weekly_retro"
      entityId={retroId}
      portfolio={portfolio}
      onCountChange={onCountChange}
    />
  );
}
