import { Platform } from 'react-native';
import type { PhotoCheck } from '@antigravity-project-spec-pack/domain/wound-model';
import { api } from './api';

/**
 * Asks the wound model about an image on this device before it is used: its quality, and whether the calibration
 * sticker is in it. Throws ApiError (status 0 when offline), so an image taken without a connection can still be used.
 */
export async function checkPhoto(uri: string): Promise<PhotoCheck> {
  const form = new FormData();
  if (Platform.OS === 'web') {
    form.append('photo', await (await fetch(uri)).blob(), 'photo.jpg');
  } else {
    // React Native's FormData sends a file from its uri.
    form.append('photo', { uri, name: 'photo.jpg', type: 'image/jpeg' } as unknown as Blob);
  }
  return api<PhotoCheck>('/model/photo-check', { method: 'POST', body: form });
}
