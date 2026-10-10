import React from 'react';
import { Image, Pressable, StyleSheet, View, useWindowDimensions } from 'react-native';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Feather from '@expo/vector-icons/Feather';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { ActionButton } from '../components/ActionButton';
import { CaptureTopBar, captureColors, captureFocus } from '../components/CaptureChrome';
import { Text } from '../components/Typography';
import { PHASE_NAME } from '../lib/format';
import { postPhoto, prePhoto } from '../lib/photos';
import { interactionStyle } from '../lib/interaction';
import { WoundDepthView, depthCamera, type DistanceReading, type WoundDepthViewRef } from '../lib/woundDepth';
import { useTreatmentContext } from '../lib/treatmentContext';
import { spacing } from '../lib/theme';
import { secureStorage } from '../store/secureStorage';
import { useVisitStore } from '../store/useVisitStore';

type ParamList = {
  Camera: { treatmentId: string; step: 'pre' | 'post' };
};

const CAPTURE_GUIDE_SEEN = 'capture-guide-seen';
const MAX_TILT_DEG = 25;

type Corner = 'topLeft' | 'topRight' | 'bottomLeft' | 'bottomRight';
const CORNERS: Corner[] = ['topLeft', 'topRight', 'bottomLeft', 'bottomRight'];

export const CameraScreen = () => {
  const navigation = useNavigation<any>();
  const route = useRoute<RouteProp<ParamList, 'Camera'>>();
  const { treatmentId, step = 'pre' } = route.params || {};
  const { treatment, patientName, visit } = useTreatmentContext(treatmentId);
  const updateTreatment = useVisitStore((state) => state.updateTreatment);
  const { width, height } = useWindowDimensions();

  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = React.useRef<CameraView>(null);
  // On an iPhone the camera also measures how far the wound is, which gives the photo its scale without a sticker.
  const depthRef = React.useRef<WoundDepthViewRef>(null);
  const [distance, setDistance] = React.useState<DistanceReading>({});

  // Last visit's photo of this wound, faintly over the viewfinder, so the new photo is taken from the same distance
  // and angle: healing is measured by comparing the two. After-cleaning photos are compared with each other.
  const previous = useVisitStore((state) =>
    state.treatments
      .filter((t) => t.caseId === treatment?.caseId && t.sequenceNumber < (treatment?.sequenceNumber ?? 0))
      .sort((a, b) => b.sequenceNumber - a.sequenceNumber)[0],
  );
  const framingUri = step === 'post' ? postPhoto(previous) : prePhoto(previous) ?? postPhoto(previous);

  // The capture guide opens by itself the first time the camera is used; after that it's behind the help button.
  React.useEffect(() => {
    let cancelled = false;
    Promise.resolve(secureStorage.getItem(CAPTURE_GUIDE_SEEN))
      .then(async (seen) => {
        if (seen || cancelled) return;
        await secureStorage.setItem(CAPTURE_GUIDE_SEEN, '1');
        if (!cancelled) navigation.navigate('CaptureGuide');
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
    // Only on opening the camera.
  }, []);

  const phase = PHASE_NAME[step];
  const title = `${phase} image`;
  const context = [patientName, visit].filter(Boolean).join(', ');
  const close = () => navigation.goBack();

  // The review screen asks the wound model whether the calibration sticker is in the image, and records the answer.
  const review = (imageUri: string) => {
    updateTreatment(treatmentId, {
      imageMetadata: {
        captureTimestamp: new Date().toISOString(),
        // Each photo's own time: a post photo taken days after the pre photo is compared with it for healing.
        takenAt: { ...treatment?.imageMetadata?.takenAt, [step]: new Date().toISOString() },
        calibrated: false,
        // A length typed in for this phase's last image belongs to that image; the other phase's stays.
        measuredLengthCm: { ...treatment?.imageMetadata?.measuredLengthCm, [step]: undefined },
      },
    });
    navigation.navigate('ImageReview', { treatmentId, step, imageUri });
  };

  // An image already on this device, e.g. one taken earlier or sent in by the patient.
  const handlePick = async () => {
    try {
      const picked = await ImagePicker.launchImageLibraryAsync({ mediaTypes: 'images', quality: 1 });
      const uri = picked.assets?.[0]?.uri;
      if (uri) review(uri);
    } catch (err) {
      console.error('Failed to choose an image', err);
    }
  };

  if (!permission) {
    return <View style={styles.screen} />;
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.screen} edges={['top', 'bottom', 'left', 'right']}>
        <CaptureTopBar
          title={title}
          context={context}
          onClose={close}
          closeLabel="Close camera"
          onHelp={() => navigation.navigate('CaptureGuide')}
        />
        <View style={styles.permission}>
          <Text style={styles.permissionTitle}>Camera access needed</Text>
          <Text style={styles.permissionBody}>
            {permission.canAskAgain
              ? 'Wound images are taken with the camera. Allow access to continue, or choose an image already on this device.'
              : "Camera access is turned off for this app. Turn it on in your device's settings, then come back, or choose an image already on this device."}
          </Text>
          <View style={styles.permissionAction}>
            {permission.canAskAgain && <ActionButton label="Allow camera access" icon="camera" onPress={requestPermission} />}
            <ActionButton label="Choose from device" icon="image" onPress={handlePick} />
          </View>
        </View>
      </SafeAreaView>
    );
  }

  const handleCapture = async () => {
    if (depthRef.current) {
      try {
        review((await depthRef.current.capture()).uri);
      } catch (err) {
        console.error('Failed to take picture', err);
      }
    } else if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync();
        if (photo) review(photo.uri);
      } catch (err) {
        console.error('Failed to take picture', err);
      }
    } else {
      // Fallback mock if camera fails
      const mockUri = `mock-camera-image-${Date.now()}`;
      navigation.navigate('ImageReview', { treatmentId, step, imageUri: mockUri });
    }
  };

  // Square guide sized to the screen, leaving room for the bars above and below.
  const guideSize = Math.round(Math.min(width - 64, height * 0.42, 320));
  // The sticker is looked for in the image itself, on the next screen: the viewfinder can't tell whether it is there.
  const framing = depthCamera
    ? 'Hold the phone square to the wound, 25 to 40 cm away, with the whole wound inside the guide.'
    : 'Frame the whole wound and the calibration sticker inside the guide.';
  const guidance = step === 'post' ? `After cleaning, before the dressing goes on. ${framing}` : framing;
  const measured = distance.distanceMm !== undefined;
  // The model corrects for tilt, but a wound seen at a slant is harder to outline, and curved skin more so.
  const slanted = (distance.tiltDeg ?? 0) > MAX_TILT_DEG;
  const guideColor = !measured ? captureColors.text : slanted ? captureColors.warning : captureColors.calibrated;
  const status = slanted
    ? { icon: 'alert-circle' as const, color: captureColors.warning, text: 'Hold the phone more square to the wound' }
    : measured
      ? { icon: 'check-circle' as const, color: captureColors.calibrated, text: `${Math.round((distance.distanceMm ?? 0) / 10)} cm from the wound` }
      : {
          icon: 'loader' as const,
          color: captureColors.text,
          text: depthCamera?.hasDepthSensor ? 'Measuring the distance' : 'Measuring the distance: move the phone slowly',
        };

  return (
    <View style={styles.screen}>
      <SafeAreaView edges={['top', 'left', 'right']}>
        <CaptureTopBar
          title={title}
          context={context}
          onClose={close}
          closeLabel="Close camera"
          onHelp={() => navigation.navigate('CaptureGuide')}
        />
      </SafeAreaView>

      <View style={styles.viewfinder}>
        {WoundDepthView ? (
          <WoundDepthView style={StyleSheet.absoluteFill} ref={depthRef} onDistance={(event) => setDistance(event.nativeEvent)} />
        ) : (
          <CameraView style={StyleSheet.absoluteFill} ref={cameraRef} facing="back" />
        )}
        {framingUri && (
          <Image
            source={{ uri: framingUri }}
            style={[StyleSheet.absoluteFill, styles.framing]}
            resizeMode="cover"
            accessibilityLabel="Last visit's photo, shown faintly to line up the new one"
          />
        )}

        <View style={styles.guideLayer}>
          <View style={{ width: guideSize, height: guideSize }}>
            {CORNERS.map((corner) => (
              <View key={corner} style={[styles.corner, styles[corner], { borderColor: guideColor }]} />
            ))}
            <View style={[styles.reticleHorizontal, { backgroundColor: guideColor }]} />
            <View style={[styles.reticleVertical, { backgroundColor: guideColor }]} />
          </View>
        </View>

        {depthCamera && (
          <View style={styles.statusLayer}>
            <View style={styles.status} accessibilityLiveRegion="polite">
              <Feather name={status.icon} size={14} color={status.color} />
              <Text style={[styles.statusText, { color: status.color }]}>{status.text}</Text>
            </View>
          </View>
        )}
      </View>

      <SafeAreaView edges={['bottom', 'left', 'right']} style={styles.controlsArea}>
        <Text style={styles.guidance}>{guidance}</Text>
        <View style={styles.controls}>
          <View style={styles.side} />
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={`Take ${phase.toLowerCase()} image`}
            onPress={handleCapture}
            style={(state) => [styles.shutter, interactionStyle(state, styles.shutterHover, styles.shutterPressed, captureFocus.ring)]}
          >
            <View style={styles.shutterCore} />
          </Pressable>
          <View style={[styles.side, styles.sideEnd]}>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel={`Choose the ${phase.toLowerCase()} image from this device`}
              onPress={handlePick}
              hitSlop={6}
              style={(state) => [styles.iconButton, interactionStyle(state, styles.iconButtonHover, styles.iconButtonPressed, captureFocus.ring)]}
            >
              <Feather name="image" size={20} color={captureColors.text} />
            </Pressable>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
};

const ARM = 28;
const STROKE = 3;

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: captureColors.background,
  },
  viewfinder: {
    flex: 1,
    overflow: 'hidden',
    backgroundColor: '#15181B',
  },

  framing: {
    opacity: 0.28,
    pointerEvents: 'none',
  },
  guideLayer: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    alignItems: 'center',
    justifyContent: 'center',
    pointerEvents: 'none',
  },
  corner: {
    position: 'absolute',
    width: ARM,
    height: ARM,
  },
  topLeft: {
    top: 0,
    left: 0,
    borderTopWidth: STROKE,
    borderLeftWidth: STROKE,
    borderTopLeftRadius: 10,
  },
  topRight: {
    top: 0,
    right: 0,
    borderTopWidth: STROKE,
    borderRightWidth: STROKE,
    borderTopRightRadius: 10,
  },
  bottomLeft: {
    bottom: 0,
    left: 0,
    borderBottomWidth: STROKE,
    borderLeftWidth: STROKE,
    borderBottomLeftRadius: 10,
  },
  bottomRight: {
    bottom: 0,
    right: 0,
    borderBottomWidth: STROKE,
    borderRightWidth: STROKE,
    borderBottomRightRadius: 10,
  },
  reticleHorizontal: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    width: 18,
    height: 1.5,
    marginLeft: -9,
    opacity: 0.8,
  },
  reticleVertical: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    width: 1.5,
    height: 18,
    marginTop: -9,
    opacity: 0.8,
  },

  statusLayer: {
    position: 'absolute',
    top: spacing.md,
    left: 0,
    right: 0,
    alignItems: 'center',
    pointerEvents: 'none',
  },
  status: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    height: 30,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(11, 13, 15, 0.72)',
  },
  statusText: {
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '600',
  },

  controlsArea: {
    paddingTop: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.md,
  },
  guidance: {
    alignSelf: 'center',
    maxWidth: 320,
    textAlign: 'center',
    fontSize: 15,
    lineHeight: 22,
    fontWeight: '500',
    color: captureColors.muted,
  },
  controls: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.md,
  },
  side: {
    flex: 1,
    alignItems: 'flex-start',
  },
  sideEnd: {
    alignItems: 'flex-end',
  },
  iconButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: captureColors.control,
  },
  iconButtonHover: {
    backgroundColor: captureColors.controlHover,
  },
  iconButtonPressed: {
    backgroundColor: captureColors.controlPressed,
  },
  // The shutter is the one full circle here: it is the camera's universal capture control.
  shutter: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: captureColors.text,
    alignItems: 'center',
    justifyContent: 'center',
  },
  shutterHover: {
    borderColor: '#FFFFFF',
  },
  shutterPressed: {
    transform: [{ scale: 0.94 }],
  },
  shutterCore: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: captureColors.text,
  },

  permission: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: spacing.lg,
  },
  permissionTitle: {
    fontSize: 22,
    lineHeight: 28,
    fontWeight: '600',
    letterSpacing: -0.4,
    color: captureColors.text,
  },
  permissionBody: {
    marginTop: 8,
    maxWidth: 360,
    fontSize: 17,
    lineHeight: 24,
    color: captureColors.muted,
  },
  permissionAction: {
    marginTop: spacing.lg,
    alignItems: 'flex-start',
    gap: spacing.sm,
  },
});
