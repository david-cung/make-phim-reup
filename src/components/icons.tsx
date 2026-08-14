import { SVGProps } from "react";

// UI redesign — inline stroke-based SVG icons.
//
// A tiny hand-picked icon set covers every button that ships with the
// editor shell. Each glyph is drawn at 24×24 in a 1.6 stroke and scales
// down cleanly to the 16–18 px the design brief calls for. We deliberately
// avoid pulling in a third-party icon library so bundle size stays
// negligible and the icons match the low-saturation, technical feel of
// the rest of the light-studio theme.
//
// Every icon takes standard SVG props so callers can style stroke color
// via `color`, size via `width/height` or a shared `.icon-16` class, and
// aria-attributes at the call site.

export type IconProps = SVGProps<SVGSVGElement> & { size?: number };

const base = (props: IconProps) => {
  const { size = 16, ...rest } = props;
  return {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.6,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    ...rest,
  };
};

export const IconMedia = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" />
  </svg>
);

export const IconSubtitles = (p: IconProps) => (
  <svg {...base(p)}>
    <rect x="3" y="5" width="18" height="14" rx="2" />
    <path d="M7 15h4M13 15h4M7 11h6M15 11h2" />
  </svg>
);

export const IconWaveform = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M3 12h2M7 8v8M11 5v14M15 8v8M19 10v4M21 12h0" />
  </svg>
);

export const IconMic = (p: IconProps) => (
  <svg {...base(p)}>
    <rect x="9" y="3" width="6" height="12" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6" />
  </svg>
);

export const IconSparkles = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2 2M16 16l2 2M18 6l-2 2M8 16l-2 2" />
  </svg>
);

export const IconSettings = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.07.31.24.6.51.82.13.11.28.19.44.24H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" />
  </svg>
);

export const IconExport = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M12 3v12M7 8l5-5 5 5M5 21h14" />
  </svg>
);

export const IconSave = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z" />
    <path d="M17 21v-8H7v8M7 3v5h8" />
  </svg>
);

export const IconUndo = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M3 7v6h6" />
    <path d="M21 17a9 9 0 0 0-15-6.7L3 13" />
  </svg>
);

export const IconRedo = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M21 7v6h-6" />
    <path d="M3 17a9 9 0 0 1 15-6.7L21 13" />
  </svg>
);

export const IconPlay = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M6 4l14 8-14 8V4Z" />
  </svg>
);

export const IconPause = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M7 4h3v16H7zM14 4h3v16h-3z" />
  </svg>
);

export const IconPlus = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M12 5v14M5 12h14" />
  </svg>
);

export const IconChevronLeft = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="m15 6-6 6 6 6" />
  </svg>
);

export const IconClose = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M6 6l12 12M18 6 6 18" />
  </svg>
);

export const IconTrash = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M4 6h16M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14" />
  </svg>
);

export const IconFilm = (p: IconProps) => (
  <svg {...base(p)}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M3 8h4M17 8h4M3 12h18M3 16h4M17 16h4" />
  </svg>
);

export const IconLayers = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="m12 3 9 5-9 5-9-5 9-5ZM3 13l9 5 9-5M3 18l9 5 9-5" />
  </svg>
);

export const IconZoomIn = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="11" cy="11" r="7" />
    <path d="M11 8v6M8 11h6M20 20l-3.5-3.5" />
  </svg>
);

export const IconZoomOut = (p: IconProps) => (
  <svg {...base(p)}>
    <circle cx="11" cy="11" r="7" />
    <path d="M8 11h6M20 20l-3.5-3.5" />
  </svg>
);

export const IconFolder = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" />
  </svg>
);

export const IconSkipBack = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M5 5v14M19 5 9 12l10 7V5Z" />
  </svg>
);

export const IconSkipForward = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M19 5v14M5 5l10 7-10 7V5Z" />
  </svg>
);

export const IconVolume = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M4 10v4h4l5 4V6L8 10H4Z" />
    <path d="M16.5 8.5a5 5 0 0 1 0 7" />
  </svg>
);

export const IconMaximize = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M8 4H4v4M16 4h4v4M8 20H4v-4M16 20h4v-4" />
  </svg>
);

export const IconCheck = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="m5 12 5 5L20 7" />
  </svg>
);

export const IconAlert = (p: IconProps) => (
  <svg {...base(p)}>
    <path d="M12 9v5M12 17h.01" />
    <path d="M10.3 4.7 2.4 18.2A1.6 1.6 0 0 0 3.8 20.5h16.4a1.6 1.6 0 0 0 1.4-2.3L13.7 4.7a1.6 1.6 0 0 0-3.4 0Z" />
  </svg>
);
