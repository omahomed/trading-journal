# Tailwind v4 / v3 Config — Decision Record

**Date:** 2026-04-30
**Phase 1 step:** 1
**Branch:** `mobile/phase-1-foundation`

## Question

Tailwind 4 (`tailwindcss@^4`, `@tailwindcss/postcss@^4`) was installed and CSS used the v4 entry `@import "tailwindcss"` — but a v3-style `frontend/tailwind.config.ts` was also present, defining the project's design tokens (colors, fonts, shadows, radii, spacing, animations). Under v4 the JS config is non-canonical (theme is normally declared via CSS `@theme`), and v4 does **not** auto-discover JS configs unless an explicit `@config "./tailwind.config.ts";` directive is present in CSS — which it was not. We needed to know empirically whether the v3 config was live, ignored, or partially read before adding any new tokens for the mobile design system.

## Probe

Added a unique sentinel color to the existing v3 config:

```ts
// frontend/tailwind.config.ts
colors: {
  "phase1-probe": "#ff00ff",
  // …existing tokens
}
```

Used the corresponding utility class in a hidden div on the dashboard page:

```tsx
// frontend/src/app/(app)/dashboard/page.tsx (TEMPORARY, removed after probe)
<div className="bg-phase1-probe" style={{ width: 1, height: 1, position: "absolute", left: -9999 }} />
```

Ran a clean production build and inspected the generated CSS chunk in `.next/static/chunks/*.css`.

## Empirical evidence

The probe color **did not appear** in the generated CSS, and neither did any other token from `tailwind.config.ts`:

```text
$ grep -c "phase1-probe" .next/static/chunks/0uzulq~dpglz0.css
0

$ grep -c "ff00ff" .next/static/chunks/0uzulq~dpglz0.css
0

$ grep -o "bg-g-dash[^{]*{[^}]*}" .next/static/chunks/0uzulq~dpglz0.css
(no output)

$ grep -o "rounded-r-3[^{]*{[^}]*}"  .next/static/chunks/0uzulq~dpglz0.css
(no output)

$ grep -o "shadow-sh-2[^{]*{[^}]*}"  .next/static/chunks/0uzulq~dpglz0.css
(no output)

$ grep -c "g-dash"   .next/static/chunks/0uzulq~dpglz0.css
0
```

The single hit for the `g-dash` color value `#6366f1` came from an inline `style={{}}` reference in the dashboard route, not from a generated utility-class rule.

A separate scan of `src/` for any usage of v3-config-derived utility classes (`bg-g-*`, `text-ink*`, `bg-bg`, `bg-surface*`, `bg-up`/`down`/`warn`, `rounded-r-1..4`, `shadow-sh-1..3`, `font-ui`/`num`/`display`, `animate-pulse-dot`/`fade-in`/`slide-up`, `w-sidebar`/`h-header`) returned **zero matches** across all `.tsx` / `.ts` files. Every token defined in `tailwind.config.ts` was simultaneously dead (not emitted) and unused (not referenced).

The styling that the app actually relies on comes from two other sources:

1. CSS custom properties defined directly in `:root` and `.dark` of `frontend/src/app/globals.css` (`--bg`, `--surface`, `--ink`, etc.).
2. Inline `style={{ background: "var(--bg)" }}` references throughout components (≈205 such usages alone for the `--font-*` family).

## Decision

**v3 config was completely ignored. Migrated all tokens to a `@theme` block in `frontend/src/app/globals.css` and deleted `frontend/tailwind.config.ts`.**

The `@theme` block defines:

- `--color-*` for `bg`/`bg-2`, `surface`/`surface-2`, `border`/`border-2`, `ink`/`ink-2..5`, `up`/`up-soft`, `down`/`down-soft`, `warn`/`warn-soft`, and the nine `g-*` group accents (`g-dash`, `g-ops`, `g-risk`, `g-daily`, `g-mkt`, `g-ai`, `g-deep`, `g-legacy`, `g-admin`) each with a `-soft` variant.
- `--font-ui`, `--font-num`, `--font-display` wired to the existing `next/font` CSS variables on `<html>` (`var(--font-inter)`, `var(--font-jetbrains)`, `var(--font-fraunces)`).
- `--radius-r-1..4`.
- `--shadow-sh-1..3`.
- `--spacing-sidebar`, `--spacing-sidebar-rail`, `--spacing-header`.
- `--animate-pulse-dot`, `--animate-fade-in`, `--animate-slide-up`, with the corresponding `@keyframes` declared just below the `@theme` block.

The pre-existing `:root` / `.dark` CSS-custom-property block in `globals.css` is left exactly as-is: it is what the inline-style code path consumes, and changing it would be a real regression risk.

This sets up Step 2 — the mobile design system can extend the same `@theme` block with `--m-*` tokens, no further config plumbing needed.

## Visual regression check

Because the v3 config was already producing zero rules and no component referenced any of those utility classes, the migration mathematically cannot change the rendering of any element on `/dashboard`, `/active-campaign`, or `/trade-journal`. The proof:

1. **Diff of built CSS** (pre-migration vs. post-migration), comparing every distinct rule body:

   ```text
   Lines added (post-only):   6
   Lines removed (pre-only):  0
   ```

   Zero existing rules were modified or removed. The 6 added lines are the 3 `@keyframes` blocks I authored (each splits across two diff segments).

2. **File-size delta:** `58,537 → 58,800 bytes` (+263 bytes), entirely accounted for by the keyframes.

3. **No emitted theme variables in the bundle:** Tailwind v4 tree-shakes `@theme` declarations whose corresponding utility classes are never used. Zero usage of `bg-g-dash`, `text-ink`, etc. → those `--color-*` vars never reach the user's browser. Existing `--bg` / `--ink` / `--surface` etc. (defined in `:root` outside `@theme`) are untouched.

4. **`npm run build`** succeeds with zero warnings vs. baseline.

5. **`npm test`** all green: 73 passed / 0 failed / 11 skipped (skipped count unchanged from baseline).

A literal pixel-comparison screenshot pass on the three pages was not performed because the underlying CSS that drives those pages (the `:root` custom properties + standard Tailwind defaults) is byte-identical pre/post — there is no rule whose value changed. If a regression were possible, it would have to come from a rule that diff confirms doesn't exist.

## Cleanup

- `frontend/tailwind.config.ts` — **deleted**.
- The temporary `phase1-probe` color and the temporary probe `<div>` in `dashboard/page.tsx` — **removed**.

## Note for future work

`frontend/src/app/(app)/layout.tsx` contains a `<style jsx global>` block that duplicates the three keyframes (`pulse-dot`, `fade-in`, `slide-up`) inline. That file is on the Phase 1 do-not-touch list, so the duplication is intentional for now. A future cleanup step (post-Phase-1) can remove the duplicate now that the keyframes live in `globals.css`.
