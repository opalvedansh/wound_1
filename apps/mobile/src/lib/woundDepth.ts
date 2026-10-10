import type { ComponentType, Ref } from 'react';
import type { ViewProps } from 'react-native';
import { requireNativeViewManager, requireOptionalNativeModule } from 'expo-modules-core';

/** How far the phone is from the wound, and what measured it. Absent until the phone has a reading. */
export interface DistanceReading {
  distanceMm?: number;
  /** How far the skin around the wound faces away from the camera, once enough of it has been measured. */
  tiltDeg?: number;
  /** lidar: the depth sensor (iPhone Pro). ar_raycast: motion tracking, on phones without one. */
  source?: 'lidar' | 'ar_raycast';
}

export interface DepthPhoto extends DistanceReading {
  uri: string;
  width: number;
  height: number;
  hfovDeg: number;
}

export interface WoundDepthViewRef {
  /** Takes the photo. The reading, when there is one, is also written into the file for the wound model. */
  capture(): Promise<DepthPhoto>;
}

type WoundDepthViewProps = ViewProps & {
  ref?: Ref<WoundDepthViewRef>;
  onDistance?: (event: { nativeEvent: DistanceReading }) => void;
};

/**
 * The iPhone camera that measures the distance to the wound (modules/wound-depth), which gives a photo its scale
 * without a calibration sticker. Null in the browser, on Android and on the simulator: there the ordinary camera
 * is used, and the size needs the sticker.
 */
const native = requireOptionalNativeModule<{ isSupported: boolean; hasDepthSensor: boolean }>('WoundDepth');

export const depthCamera = native?.isSupported ? { hasDepthSensor: native.hasDepthSensor } : null;

export const WoundDepthView: ComponentType<WoundDepthViewProps> | null = depthCamera
  ? requireNativeViewManager<WoundDepthViewProps>('WoundDepth')
  : null;
