import React from 'react';
import { Image, Pressable, StyleSheet, TextInput, View } from 'react-native';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Feather from '@expo/vector-icons/Feather';
import { ActionButton } from '../components/ActionButton';
import { CaptureTopBar, captureColors, captureFocus } from '../components/CaptureChrome';
import { Text } from '../components/Typography';
import { PHASE_NAME } from '../lib/format';
import { interactionStyle } from '../lib/interaction';
import type { PhotoCheck } from '@antigravity-project-spec-pack/domain/wound-model';
import { ApiError } from '../lib/api';
import { checkPhoto } from '../lib/photoCheck';
import { supabase } from '../lib/supabase';
import { radii, spacing } from '../lib/theme';
import { useTreatmentContext } from '../lib/treatmentContext';
import { useVisitStore } from '../store/useVisitStore';

type ParamList = {
  ImageReview: { treatmentId: string; step: 'pre' | 'post'; imageUri: string };
};

type FeatherName = React.ComponentProps<typeof Feather>['name'];

interface StatusRowProps {
  icon: FeatherName;
  color: string;
  title: string;
  detail?: string;
}

/**
 * What the wound model said about the image. `refused`: the server won't take the file (not an image, too large).
 * `unavailable`: no answer (offline, or the model is down), so the image is checked when it is uploaded instead.
 */
type Check = 'checking' | 'unavailable' | { refused: string } | PhotoCheck;

// What a ruler can give for a wound's length (the model service refuses anything outside this).
const MIN_LENGTH_CM = 0.2;
const MAX_LENGTH_CM = 60;

// An image on this device. Anything else was checked when it was first taken (see `carriedFrom`).
const isLocal = (uri?: string) => !!uri && /^(file:|data:|blob:)/.test(uri);

export const ImageReviewScreen = () => {
  const navigation = useNavigation<any>();
  const route = useRoute<RouteProp<ParamList, 'ImageReview'>>();
  const { treatmentId, step = 'pre', imageUri } = route.params || {};
  const updateTreatment = useVisitStore((state) => state.updateTreatment);
  const { treatment, patientName, visit } = useTreatmentContext(treatmentId);
  const previous = useVisitStore((state) =>
    treatment
      ? state.treatments.find((t) => t.caseId === treatment.caseId && t.sequenceNumber === treatment.sequenceNumber - 1)
      : undefined,
  );

  const [check, setCheck] = React.useState<Check | null>(isLocal(imageUri) ? 'checking' : null);
  const [displayUri, setDisplayUri] = React.useState<string | null>(null);
  // The wound's longest length by ruler, for an image with nothing in it to measure with.
  const [length, setLength] = React.useState('');

  React.useEffect(() => {
    if (imageUri) {
      // Anything else is a storage path. On the web a captured or chosen image is a data: or blob: address.
      if (/^(file:|mock-|http|data:|blob:)/.test(imageUri)) {
        setDisplayUri(imageUri);
      } else {
        const fetchSignedUrl = async () => {
          const { data, error } = await supabase.storage.from('images').createSignedUrl(imageUri, 3600);
          if (data?.signedUrl) {
            setDisplayUri(data.signedUrl);
          } else {
            console.error('Error fetching signed URL:', error);
            setDisplayUri(imageUri);
          }
        };
        fetchSignedUrl();
      }
    }
  }, [imageUri]);

  React.useEffect(() => {
    if (!isLocal(imageUri)) return;
    let cancelled = false;
    setCheck('checking');
    checkPhoto(imageUri).then(
      (result) => {
        if (cancelled) return;
        setCheck(result);
        updateTreatment(treatmentId, {
          imageMetadata: {
            ...treatment?.imageMetadata,
            captureTimestamp: treatment?.imageMetadata?.captureTimestamp ?? new Date().toISOString(),
            calibrated: result.marker_found || !!result.phone_reading,
          },
        });
      },
      (error) => {
        if (cancelled) return;
        const refused = error instanceof ApiError && (error.status === 400 || error.status === 413);
        setCheck(refused ? { refused: error.message } : 'unavailable');
      },
    );
    return () => {
      cancelled = true;
    };
    // Once per image: the treatment changes as the result is stored.
  }, [imageUri]);

  const result = check && typeof check === 'object' && 'quality' in check ? check : null;
  const refused = check && typeof check === 'object' && 'refused' in check ? check.refused : undefined;
  const checking = check === 'checking';
  const failed = refused !== undefined || result?.quality.usable === false;

  // Nothing in the image gives it a scale, so the clinician's own measurement can.
  // (Unchecked images too: the model uses the length only if it finds no sticker or phone reading.)
  const needsLength = check === 'unavailable' || (!!result && !failed && !result.marker_found && !result.phone_reading);
  const lengthCm = Number(length.replace(',', '.'));
  const lengthGiven = needsLength && length.trim() !== '';
  const lengthValid = Number.isFinite(lengthCm) && lengthCm >= MIN_LENGTH_CM && lengthCm <= MAX_LENGTH_CM;

  const handleApprove = () => {
    if (failed) {
      alert("This image can't be used. Please retake.");
      return;
    }
    if (lengthGiven && !lengthValid) {
      alert(`Enter the wound's length in centimetres, between ${MIN_LENGTH_CM} and ${MAX_LENGTH_CM}, or leave it empty.`);
      return;
    }
    if (treatment?.imageMetadata) {
      updateTreatment(treatmentId, {
        imageMetadata: {
          ...treatment.imageMetadata,
          measuredLengthCm: { ...treatment.imageMetadata.measuredLengthCm, [step]: lengthGiven ? lengthCm : undefined },
        },
      });
    }

    if (step === 'pre') {
      updateTreatment(treatmentId, { preImageUri: imageUri });
      navigation.navigate('ClinicalAssessment', { treatmentId });
    } else {
      updateTreatment(treatmentId, { postImageUri: imageUri });
      navigation.navigate('TherapyTracking', { treatmentId });
    }
  };

  const phase = PHASE_NAME[step];
  const context = [patientName, visit].filter(Boolean).join(', ');
  // A new treatment's pre-treatment image starts as the previous treatment's post-treatment image (see addTreatment).
  const carriedFrom =
    step === 'pre' && !!imageUri && imageUri === previous?.postImageUri ? previous.sequenceNumber : undefined;

  // The model's own sentences say what is wrong and what to do next time, so they are shown as they are.
  const quality: StatusRowProps | null = checking
    ? { icon: 'loader', color: captureColors.text, title: 'Checking the image' }
    : check === 'unavailable'
      ? {
          icon: 'alert-circle',
          color: captureColors.muted,
          title: "Couldn't check this image now",
          detail: 'It will be checked when it is uploaded.',
        }
      : failed
        ? { icon: 'alert-triangle', color: captureColors.warning, title: 'Retake needed', detail: refused ?? result?.quality.issues[0] }
        : result?.quality.issues.length
          ? {
              icon: 'alert-circle',
              color: captureColors.warning,
              title: 'Usable, but check the result against the image',
              detail: result.quality.issues.join(' '),
            }
          : result
            ? { icon: 'check-circle', color: captureColors.calibrated, title: 'Image quality OK' }
            : null;

  // The sticker, or the distance the phone measured, is what gives the image a scale. Without either the wound
  // is still outlined and assessed.
  const calibration: StatusRowProps | null =
    !result || failed
      ? null
      : result.marker_found
        ? {
            icon: 'check-circle',
            color: captureColors.calibrated,
            title: 'Calibration sticker found',
            detail: "The wound's size will be measured.",
          }
        : result.phone_reading
          ? {
              icon: 'check-circle',
              color: captureColors.calibrated,
              title: `Distance measured: ${Math.round(result.phone_reading.distance_mm / 10)} cm`,
              detail: result.phone_reading.normal
                ? "The wound's size will be measured from it, corrected for the phone's tilt."
                : "The wound's size will be measured from it. It is right only if the phone was held square to the wound.",
            }
          : {
              icon: 'alert-circle',
              color: captureColors.warning,
              title: 'No calibration sticker in this image',
              detail:
                "The wound will still be outlined and assessed. To get its size, place the sticker beside the wound and retake, or enter the wound's length below.",
            };

  return (
    <View style={styles.screen}>
      <SafeAreaView edges={['top', 'left', 'right']}>
        <CaptureTopBar
          title={`${phase} image`}
          context={context}
          onClose={() => navigation.goBack()}
          closeLabel="Back to camera"
          closeIcon="chevron-left"
        />
      </SafeAreaView>

      <View style={styles.imageArea}>
        {displayUri && !displayUri.startsWith('mock') ? (
          <Image
            source={{ uri: displayUri }}
            style={styles.image}
            resizeMode="contain"
            accessibilityLabel={`${phase} image`}
            accessibilityIgnoresInvertColors
          />
        ) : (
          <Text style={styles.noPreview}>{displayUri ? 'No preview for this image' : 'Loading image'}</Text>
        )}
      </View>

      <SafeAreaView edges={['bottom', 'left', 'right']} style={styles.panel}>
        {carriedFrom !== undefined && (
          <Text style={styles.note}>Carried forward from treatment {carriedFrom}'s post-treatment image.</Text>
        )}
        <View style={styles.statusList} accessibilityLiveRegion="polite">
          {quality && <StatusRow {...quality} />}
          {calibration && <StatusRow {...calibration} />}
        </View>
        {needsLength && (
          <View style={styles.lengthField}>
            <Text style={styles.lengthLabel} nativeID="wound-length-label">
              Wound length by ruler, in cm (optional)
            </Text>
            <TextInput
              value={length}
              onChangeText={setLength}
              keyboardType="decimal-pad"
              inputMode="decimal"
              placeholder="e.g. 3.2"
              placeholderTextColor={captureColors.muted}
              accessibilityLabelledBy="wound-length-label"
              aria-labelledby="wound-length-label"
              style={[styles.lengthInput, lengthGiven && !lengthValid && styles.lengthInputInvalid]}
            />
            <Text style={styles.statusDetail}>
              Measure the wound's longest side. The size is then worked out from that and the outline.
            </Text>
          </View>
        )}
        <View style={styles.actions}>
          <Pressable
            accessibilityRole="button"
            onPress={() => navigation.goBack()}
            style={(state) => [
              styles.secondary,
              interactionStyle(state, styles.secondaryHover, styles.secondaryPressed, captureFocus.ring),
            ]}
          >
            <Feather name="rotate-ccw" size={18} color={captureColors.text} />
            <Text style={styles.secondaryLabel}>Retake</Text>
          </Pressable>
          <ActionButton
            label="Use image"
            icon="check"
            onPress={handleApprove}
            disabled={checking || failed}
            style={styles.primary}
          />
        </View>
      </SafeAreaView>
    </View>
  );
};

const StatusRow = ({ icon, color, title, detail }: StatusRowProps) => (
  <View style={styles.statusRow}>
    <Feather name={icon} size={16} color={color} style={styles.statusIcon} />
    <View style={styles.statusBody}>
      <Text style={[styles.statusTitle, { color }]}>{title}</Text>
      {!!detail && <Text style={styles.statusDetail}>{detail}</Text>}
    </View>
  </View>
);

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: captureColors.background,
  },
  imageArea: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#15181B',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  noPreview: {
    fontSize: 15,
    lineHeight: 22,
    fontWeight: '500',
    color: captureColors.muted,
  },

  panel: {
    paddingTop: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.md,
  },
  note: {
    marginBottom: 12,
    fontSize: 13,
    lineHeight: 18,
    color: captureColors.muted,
  },
  statusList: {
    gap: 12,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
  },
  statusIcon: {
    marginTop: 2,
  },
  statusBody: {
    flex: 1,
  },
  statusTitle: {
    fontSize: 15,
    lineHeight: 20,
    fontWeight: '600',
  },
  statusDetail: {
    marginTop: 2,
    fontSize: 14,
    lineHeight: 20,
    color: captureColors.muted,
  },

  lengthField: {
    marginTop: spacing.md,
    gap: 6,
  },
  lengthLabel: {
    fontSize: 15,
    lineHeight: 20,
    fontWeight: '600',
    color: captureColors.text,
  },
  lengthInput: {
    height: 44,
    paddingHorizontal: 12,
    borderRadius: radii.control,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.28)',
    fontSize: 17,
    color: captureColors.text,
    backgroundColor: captureColors.control,
  },
  lengthInputInvalid: {
    borderColor: captureColors.warning,
  },

  actions: {
    flexDirection: 'row',
    gap: 12,
    marginTop: spacing.lg,
  },
  secondary: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    height: 52,
    borderRadius: radii.control,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.28)',
  },
  secondaryHover: {
    backgroundColor: captureColors.control,
  },
  secondaryPressed: {
    backgroundColor: captureColors.controlPressed,
  },
  secondaryLabel: {
    fontSize: 16,
    lineHeight: 20,
    fontWeight: '600',
    color: captureColors.text,
  },
  primary: {
    flex: 1,
    height: 52,
  },
});
