import React, { useState } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import Feather from '@expo/vector-icons/Feather';
import {
  questionsFor,
  type Question,
  type QuestionForm,
  type QuestionResponse,
} from '@antigravity-project-spec-pack/domain/questions';
import { FactStrip, type Fact } from '../components/FactStrip';
import { SyncLabel } from '../components/SyncLabel';
import { Text } from '../components/Typography';
import {
  PHASE_NAME,
  TREND_COLOR,
  calendarDate,
  dayDiff,
  parseDate,
  plural,
  questionTitle,
  sentenceCase,
  timeAgo,
} from '../lib/format';
import { interactionStyle } from '../lib/interaction';
import { postLabel, postPhoto, prePhoto, woundAnalysis } from '../lib/photos';
import { answerText } from '../lib/questionAnswers';
import { useQuestionCatalog } from '../lib/questionCatalog';
import { useStalledSync } from '../lib/syncStatus';
import { colors, radii, spacing } from '../lib/theme';
import { useTreatmentContext } from '../lib/treatmentContext';
import { useVisitStore } from '../store/useVisitStore';

type ParamList = {
  TreatmentDetail: { treatmentId: string };
};

type Phase = keyof typeof PHASE_NAME;

interface Detail {
  label: string;
  value?: string;
}

const CONTENT_MAX_WIDTH = 680;

const tone = {
  tile: '#F1F3F5',
};

/** The healing verdict's colour: the same three the clinician's own trend uses. */
const VERDICT_COLOR = {
  improving: TREND_COLOR.Improving,
  static: TREND_COLOR.Static,
  deteriorating: TREND_COLOR.Deteriorating,
  baseline: colors.textSecondary,
  not_compared: colors.textSecondary,
} as const;

/**
 * A form's answers on this record, as label and value rows. Built-in answers are labelled with the question
 * as it's worded now; answers to added questions keep the wording they were given under. A hidden
 * built-in question only shows up when this record answered it. `skip` leaves out fields shown elsewhere.
 */
const recordedAnswers = (
  catalog: Question[],
  form: QuestionForm,
  record: object | undefined,
  responses: QuestionResponse[] | undefined,
  skip: string[],
): Detail[] => {
  const fields = (record ?? {}) as Record<string, unknown>;
  return [
    ...questionsFor(catalog, form).flatMap((question) => {
      if (!question.fieldKey || skip.includes(question.fieldKey)) return [];
      const value = answerText(question.type, fields[question.fieldKey]);
      return question.active || value ? [{ label: questionTitle(question.title), value }] : [];
    }),
    ...(responses ?? []).map((response) => ({
      label: questionTitle(response.title),
      value: answerText(response.type, response.answer),
    })),
  ];
};

/**
 * The next visit as the therapy form records it: an interval ("1 Week", "In 3 Days"), "PRN (As Needed)",
 * or a date. Intervals stay anchored to this visit's date instead of becoming a calculated calendar date.
 */
const nextVisitFact = (recorded: string | undefined, visitDate: Date | null, now: Date): Pick<Fact, 'value' | 'detail'> => {
  const text = recorded?.trim();
  if (!text) return { value: 'Not set' };
  if (/^prn\b/i.test(text)) return { value: 'As needed', detail: 'PRN' };
  const interval = text.match(/^(?:in\s+)?(\d+)\s*(day|week|month)s?$/i);
  if (interval) {
    return {
      value: plural(Number(interval[1]), interval[2].toLowerCase()),
      detail: visitDate ? `after ${calendarDate(visitDate, now)}` : 'after this visit',
    };
  }
  const date = parseDate(text);
  if (!date) return { value: text };
  const days = dayDiff(now, date);
  return {
    value: calendarDate(date, now),
    detail: days < 0 ? timeAgo(date, now) : days === 0 ? 'Today' : days === 1 ? 'Tomorrow' : `In ${plural(days, 'day')}`,
  };
};

export const TreatmentDetailScreen = () => {
  const navigation = useNavigation<any>();
  const route = useRoute<RouteProp<ParamList, 'TreatmentDetail'>>();
  const insets = useSafeAreaInsets();
  const treatmentId = route.params?.treatmentId;
  const { treatment, woundCase, patientName } = useTreatmentContext(treatmentId);
  const previous = useVisitStore((state) =>
    treatment
      ? state.treatments.find((t) => t.caseId === treatment.caseId && t.sequenceNumber === treatment.sequenceNumber - 1)
      : undefined,
  );
  // Treatments have no sync state of their own; one is waiting while it's still in the outbox.
  const queued = useVisitStore((state) => !!state.outbox.treatments[treatmentId]);
  const stalled = useStalledSync();
  const catalog = useQuestionCatalog((state) => state.questions);

  const backButton = (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel="Back"
      hitSlop={6}
      onPress={() => navigation.goBack()}
      style={(state) => [styles.iconButton, interactionStyle(state, styles.iconButtonHover, styles.iconButtonPressed)]}
    >
      <Feather name="chevron-left" size={20} color={colors.textPrimary} />
    </Pressable>
  );

  if (!treatment) {
    return (
      <SafeAreaView style={styles.safe} edges={['top', 'left', 'right']}>
        <View style={styles.page}>
          <View style={styles.column}>
            <View style={styles.topBar}>{backButton}</View>
            <View style={styles.notFound}>
              <Text style={styles.notFoundTitle}>Treatment not found</Text>
              <Text style={styles.notFoundBody}>This treatment isn't on this device.</Text>
            </View>
          </View>
        </View>
      </SafeAreaView>
    );
  }

  const now = new Date();
  const visitDate = parseDate(treatment.createdAt);
  const done = treatment.phase === 'COMPLETED';
  // A new treatment starts with a copy of the previous assessment. It only becomes this visit's
  // record when the clinician saves the assessment, which moves the treatment on from PRE.
  const assessment = treatment.phase === 'PRE' ? undefined : treatment.assessment;
  const trend = assessment?.woundAppearanceTrend;
  const therapy = treatment.therapy;
  // The pre-treatment image starts as the previous treatment's post-treatment image (see addTreatment).
  const carriedFrom =
    treatment.preImageUri && treatment.preImageUri === previous?.postImageUri ? previous.sequenceNumber : undefined;
  const sync = stalled.ids.has(treatment.id) ? 'failed' : queued ? 'pending' : 'synced';

  const facts: Fact[] = [
    {
      label: 'Status',
      value: done ? 'Completed' : 'In progress',
      detail: trend,
      detailColor: trend ? TREND_COLOR[trend] : undefined,
      flex: 1.15,
    },
    {
      label: 'Date',
      value: visitDate ? calendarDate(visitDate, now) : 'Not recorded',
      detail: visitDate ? timeAgo(visitDate, now) : undefined,
    },
    { label: 'Next visit', ...nextVisitFact(therapy?.nextVisitDate, visitDate, now) },
  ];

  // The next visit and the trend are already in the fact strip.
  const care = recordedAnswers(catalog, 'care', therapy, therapy?.responses, ['nextVisitDate']);
  const findings = recordedAnswers(catalog, 'assessment', assessment, assessment?.responses, ['woundAppearanceTrend']);
  const analysis = woundAnalysis(treatment);

  return (
    <SafeAreaView style={styles.safe} edges={['top', 'left', 'right']}>
      <ScrollView
        contentContainerStyle={[styles.page, { paddingBottom: insets.bottom + spacing.xxl }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.column}>
          <View style={styles.topBar}>
            {backButton}
            <SyncLabel state={sync} />
          </View>

          <Text style={styles.context}>
            {patientName !== undefined && <Text style={styles.contextName}>{patientName}</Text>}
            {patientName !== undefined && woundCase ? ', ' : ''}
            {woundCase ? sentenceCase(woundCase.woundLocation) : ''}
          </Text>
          <Text variant="h2" style={styles.title} accessibilityRole="header">
            Treatment {treatment.sequenceNumber}
          </Text>
          <Text style={styles.subtitle}>{treatment.sequenceNumber === 1 ? 'Baseline visit' : 'Follow-up visit'}</Text>

          <FactStrip facts={facts} style={styles.facts} />

          <SectionHeader title="Before and after" />
          <View style={styles.phases}>
            <PhaseImage
              phase="pre"
              uri={prePhoto(treatment)}
              due={treatment.phase === 'PRE'}
              note={carriedFrom !== undefined ? `Carried forward from treatment ${carriedFrom}` : undefined}
            />
            <PhaseImage phase="post" uri={postPhoto(treatment)} due={treatment.phase === 'POST'} note={postLabel(treatment)} />
          </View>

          {analysis && (
            <>
              <SectionHeader title="Wound analysis" note="AI draft for a clinician to review" />
              {analysis.notice !== undefined ? (
                <Text style={styles.absent}>{analysis.notice}</Text>
              ) : (
                <>
                  {analysis.flags.map((flag) => (
                    <View key={flag.text} style={[styles.flag, flag.level === 'urgent' && styles.flagUrgent]}>
                      <Text style={[styles.flagText, flag.level === 'urgent' && styles.flagTextUrgent]}>
                        <Text style={styles.flagLevel}>{flag.level === 'urgent' ? 'Urgent: ' : 'Review: '}</Text>
                        {flag.text}
                      </Text>
                    </View>
                  ))}

                  <Text style={styles.groupTitle}>{PHASE_NAME.pre} photo</Text>
                  <DetailList rows={analysis.before} />

                  {analysis.after && (
                    <>
                      <Text style={styles.groupTitle}>{PHASE_NAME.post} photo ({analysis.after.label.toLowerCase()})</Text>
                      <DetailList rows={analysis.after.facts} />
                    </>
                  )}

                  {analysis.healing && (
                    <>
                      <Text style={styles.groupTitle}>Healing</Text>
                      <View style={styles.verdict}>
                        <Text style={[styles.verdictTitle, { color: VERDICT_COLOR[analysis.healing.state] }]}>{analysis.healing.title}</Text>
                        <Text style={styles.verdictDetail}>{analysis.healing.detail}</Text>
                      </View>
                      {analysis.healing.rows.length > 0 && <DetailList rows={analysis.healing.rows} />}
                    </>
                  )}

                  <Text style={styles.groupTitle}>Review</Text>
                  <DetailList rows={analysis.status} />
                </>
              )}
            </>
          )}

          <SectionHeader title="Care provided" />
          {therapy ? (
            <DetailList rows={care} />
          ) : (
            <Text style={styles.absent}>
              {done ? 'No care was recorded for this treatment.' : 'Recorded after the post-treatment image.'}
            </Text>
          )}

          <SectionHeader title="Assessment" note="Before treatment" />
          {assessment ? (
            <DetailList rows={findings} />
          ) : (
            <Text style={styles.absent}>
              {treatment.phase === 'PRE'
                ? 'Recorded after the pre-treatment image.'
                : 'No assessment was recorded for this treatment.'}
            </Text>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const SectionHeader = ({ title, note }: { title: string; note?: string }) => (
  <View style={styles.sectionHeader}>
    <Text style={styles.sectionTitle} accessibilityRole="header">
      {title}
    </Text>
    {note !== undefined && <Text style={styles.sectionNote}>{note}</Text>}
  </View>
);

/** One phase's image with its name underneath, so pre and post are never confused. */
const PhaseImage = ({ phase, uri, due, note }: { phase: Phase; uri?: string; due: boolean; note?: string }) => {
  const [failed, setFailed] = useState(false);
  const label = PHASE_NAME[phase];
  // Mock captures and images that fail to load were still taken, so they read "No preview", not "Not captured".
  const viewable = !!uri && !failed && !uri.startsWith('mock');
  return (
    <View style={styles.phase}>
      {viewable ? (
        <Image
          source={{ uri }}
          style={styles.phaseImage}
          onError={() => setFailed(true)}
          accessibilityLabel={`${label} image`}
          accessibilityIgnoresInvertColors
        />
      ) : (
        <View style={[styles.phaseImage, styles.phaseEmpty]}>
          <Text style={[styles.phaseEmptyText, !uri && due && styles.phaseDue]}>
            {uri ? 'No preview' : due ? 'Due' : 'Not captured'}
          </Text>
        </View>
      )}
      <Text style={styles.phaseLabel}>{label}</Text>
      {note !== undefined && <Text style={styles.phaseNote}>{note}</Text>}
    </View>
  );
};

/** Label and value rows, like the findings section of a chart. Gaps read "Not recorded" rather than disappearing. */
const DetailList = ({ rows }: { rows: Detail[] }) => (
  <View style={styles.list}>
    {rows.map((row, i) => (
      <View key={row.label} style={[styles.listRow, i > 0 && styles.listDivider]}>
        <Text style={styles.listLabel}>{row.label}</Text>
        <Text style={[styles.listValue, !row.value && styles.listMissing]}>{row.value || 'Not recorded'}</Text>
      </View>
    ))}
  </View>
);

const hairline = StyleSheet.hairlineWidth;

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.background,
  },
  page: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
  },
  column: {
    width: '100%',
    maxWidth: CONTENT_MAX_WIDTH,
    alignSelf: 'center',
  },

  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: 48,
    paddingTop: 12,
  },
  iconButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  iconButtonHover: {
    backgroundColor: colors.surfaceHover,
  },
  iconButtonPressed: {
    backgroundColor: colors.surfacePressed,
  },

  context: {
    marginTop: spacing.lg,
    fontSize: 15,
    lineHeight: 20,
    color: colors.textSecondary,
  },
  contextName: {
    fontWeight: '600',
    color: colors.textPrimary,
  },
  title: {
    marginTop: 4,
    lineHeight: 34,
  },
  subtitle: {
    marginTop: 2,
    fontSize: 17,
    lineHeight: 24,
    fontWeight: '500',
    color: colors.textSecondary,
  },
  facts: {
    marginTop: 20,
  },

  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
    marginTop: 36,
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 20,
    lineHeight: 26,
    fontWeight: '700',
    letterSpacing: -0.3,
    color: colors.textPrimary,
  },
  sectionNote: {
    fontSize: 15,
    lineHeight: 20,
    fontWeight: '500',
    color: colors.textMuted,
  },

  // Square tiles match the square capture guide, so the framed wound fills the crop.
  phases: {
    flexDirection: 'row',
    gap: 12,
  },
  phase: {
    flex: 1,
  },
  phaseImage: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: radii.control,
    overflow: 'hidden',
    backgroundColor: tone.tile,
  },
  phaseEmpty: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  phaseEmptyText: {
    fontSize: 14,
    lineHeight: 20,
    fontWeight: '500',
    color: colors.textMuted,
  },
  phaseDue: {
    fontWeight: '600',
    color: colors.pending,
  },
  phaseLabel: {
    marginTop: 8,
    fontSize: 15,
    lineHeight: 20,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  phaseNote: {
    marginTop: 2,
    fontSize: 13,
    lineHeight: 18,
    color: colors.textMuted,
  },

  list: {
    borderRadius: radii.control,
    borderWidth: hairline,
    borderColor: colors.border,
    backgroundColor: colors.surface,
  },
  // Label above value: question titles come from the admin-edited catalog and can be long.
  listRow: {
    gap: 2,
    paddingVertical: 11,
    paddingHorizontal: spacing.md,
  },
  listDivider: {
    borderTopWidth: hairline,
    borderTopColor: colors.border,
  },
  listLabel: {
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '500',
    color: colors.textMuted,
  },
  listValue: {
    fontSize: 15,
    lineHeight: 22,
    fontWeight: '500',
    color: colors.textPrimary,
  },
  listMissing: {
    fontWeight: '400',
    color: colors.textMuted,
  },
  absent: {
    fontSize: 15,
    lineHeight: 22,
    color: colors.textMuted,
  },

  groupTitle: {
    marginTop: 20,
    marginBottom: 8,
    fontSize: 15,
    lineHeight: 20,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  flag: {
    marginBottom: 8,
    paddingVertical: 10,
    paddingHorizontal: spacing.md,
    borderRadius: radii.control,
    borderWidth: hairline,
    borderColor: '#FCD34D',
    backgroundColor: '#FFFBEB',
  },
  flagUrgent: {
    borderColor: '#FCA5A5',
    backgroundColor: '#FEF2F2',
  },
  flagText: {
    fontSize: 15,
    lineHeight: 22,
    color: '#78350F',
  },
  flagTextUrgent: {
    color: '#7F1D1D',
  },
  flagLevel: {
    fontWeight: '700',
  },
  verdict: {
    marginBottom: 8,
  },
  verdictTitle: {
    fontSize: 20,
    lineHeight: 26,
    fontWeight: '700',
    letterSpacing: -0.3,
  },
  verdictDetail: {
    marginTop: 2,
    fontSize: 14,
    lineHeight: 20,
    color: colors.textMuted,
  },

  notFound: {
    marginTop: spacing.xl,
  },
  notFoundTitle: {
    fontSize: 22,
    lineHeight: 28,
    fontWeight: '600',
    letterSpacing: -0.4,
    color: colors.textPrimary,
  },
  notFoundBody: {
    marginTop: 6,
    fontSize: 17,
    lineHeight: 24,
    color: colors.textSecondary,
  },
});
