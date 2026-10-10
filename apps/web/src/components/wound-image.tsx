'use client';

import { ImageOff } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { cx } from './ui';

type Outline = [number, number][][];

/**
 * A wound photo (signed URL from the API, cached by the browser) with the model's outline drawn over it.
 * The outline's points are 0–1 of the photo, so the overlay uses the photo's own size and the same
 * "cover" cropping as the image: it stays on the wound at any aspect ratio.
 */
export function WoundImage({
  src,
  outline,
  aspect = 3 / 4,
  className,
  label = 'Wound photo',
  eager = false,
}: {
  src: string | null | undefined;
  outline?: Outline | null;
  /** Height ÷ width of the frame. */
  aspect?: number;
  className?: string;
  label?: string;
  /** Load at once (the page's main photo) instead of when scrolled into view. */
  eager?: boolean;
}) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [failed, setFailed] = useState(false);
  const img = useRef<HTMLImageElement>(null);
  // A cached photo can finish loading before React hydrates, so onLoad never fires and the outline would never
  // be drawn: read the size from an image that is already complete.
  useEffect(() => {
    const el = img.current;
    if (el?.complete && el.naturalWidth) setSize({ w: el.naturalWidth, h: el.naturalHeight });
  }, [src]);

  if (!src || failed) {
    return (
      <div
        className={cx('grid place-items-center bg-surface-alt text-[12px] text-faint', className)}
        style={{ aspectRatio: `${1 / aspect}` }}
      >
        <span className="flex flex-col items-center gap-1">
          <ImageOff size={18} />
          {failed ? 'Photo unavailable' : 'No photo'}
        </span>
      </div>
    );
  }

  return (
    <div className={cx('relative overflow-hidden bg-surface-alt', className)} style={{ aspectRatio: `${1 / aspect}` }}>
      {/* eslint-disable-next-line @next/next/no-img-element -- signed storage URL, already resized */}
      <img
        ref={img}
        src={src}
        alt={label}
        loading={eager ? 'eager' : 'lazy'}
        decoding="async"
        onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        onError={() => setFailed(true)}
        className="absolute inset-0 size-full object-cover"
      />
      {outline?.length && size ? (
        <svg
          viewBox={`0 0 ${size.w} ${size.h}`}
          preserveAspectRatio="xMidYMid slice"
          className="absolute inset-0 size-full"
          aria-hidden
        >
          {outline.map((poly, i) => (
            <path
              key={i}
              d={poly.map(([x, y], j) => `${j ? 'L' : 'M'}${(x * size.w).toFixed(1)} ${(y * size.h).toFixed(1)}`).join(' ') + ' Z'}
              fill="rgba(67,185,169,0.16)"
              stroke="#43E0C8"
              strokeWidth={2}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      ) : null}
    </div>
  );
}
