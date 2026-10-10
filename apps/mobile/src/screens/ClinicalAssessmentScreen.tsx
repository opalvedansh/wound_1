import React, { useEffect, useRef, useState } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import type { QuestionAnswer } from '@antigravity-project-spec-pack/domain/questions';
import { FormField, TextField } from '../components/Form';
import { FormLayout } from '../components/FormLayout';
import { QuestionField } from '../components/QuestionField';
import { Text } from '../components/Typography';
import { PHASE_NAME, TREND_COLOR, type Trend } from '../lib/format';
import { askedQuestions, useQuestionCatalog } from '../lib/questionCatalog';
import {
  answerFor,
  answerKey,
  answersFrom,
  fieldAnswer,
  isAnswered,
  listAnswer,
  responsesFrom,
  textAnswer,
  type Answers,
} from '../lib/questionAnswers';
import { colors, radii, spacing } from '../lib/theme';
import { useTreatmentContext } from '../lib/treatmentContext';
import { useVisitStore } from '../store/useVisitStore';

type ParamList = {
  ClinicalAssessment: { treatmentId: string };
};

const PHASE_LABEL = {
  PRE: PHASE_NAME.pre,
  POST: PHASE_NAME.post,
  COMPLETED: 'Completed',
} as const;

/** Depth in cm by probe. A photo cannot show it, so it is the one finding only the clinician can give. */
const MAX_DEPTH_CM = 30;

const isTrend = (value: QuestionAnswer | undefined): value is Trend => typeof value === 'string' && value in TREND_COLOR;

export const ClinicalAssessmentScreen = () => {
  const navigation = useNavigation<any>();
  const route = useRoute<RouteProp<ParamList, 'ClinicalAssessment'>>();
  const treatmentId = route.params?.treatmentId;
  const { treatment, patientName, visit } = useTreatmentContext(treatmentId);
  const updateTreatment = useVisitStore((state) => state.updateTreatment);
  const completePrePhase = useVisitStore((state) => state.completePrePhase);
  const catalog = useQuestionCatalog((state) => state.questions);
  const refreshCatalog = useQuestionCatalog((state) => state.refresh);
  const isFollowUp = (treatment?.sequenceNumber || 1) > 1;

  useEffect(() => {
    refreshCatalog();
  }, [refreshCatalog]);

  // Starts from what's on the record; a follow-up treatment starts with a copy of the last assessment.
  const [answers, setAnswers] = useState<Answers>(() => ({
    // The pain score has always started at 0.
    pain: 0,
    ...answersFrom(treatment?.assessment, treatment?.assessment?.responses),
  }));
  // Depth is probed afresh at every visit, so a new treatment doesn't start with the last visit's.
  const [depth, setDepth] = useState(() =>
    treatment && treatment.phase !== 'PRE' && treatment.assessment?.depthCm !== undefined ? String(treatment.assessment.depthCm) : '',
  );
  const [submitted, setSubmitted] = useState(false);
  const depthOffset = useRef<number | undefined>(undefined);
  const scrollRef = useRef<React.ComponentRef<typeof ScrollView>>(null);
  const questionOffsets = useRef<Record<string, number>>({});

  const asked = askedQuestions(catalog, 'assessment', isFollowUp);
  // Only questions on screen can block saving; a 0 on a numeric question counts as an answer.
  const unanswered = asked.filter((q) => q.required && !isAnswered(q, answers));

  if (!treatment || !treatmentId) {
    return (
      <FormLayout mode="step" title="Clinical assessment" onClose={() => navigation.goBack()}>
        <View style={styles.notFound}>
          <Text style={styles.notFoundTitle}>Treatment not found</Text>
          <Text style={styles.notFoundBody}>Go back to the case and open the treatment again.</Text>
        </View>
      </FormLayout>
    );
  }

  const setAnswer = (key: string, value: QuestionAnswer | undefined) => setAnswers((prev) => ({ ...prev, [key]: value }));

  const depthCm = Number(depth.replace(',', '.'));
  const depthGiven = depth.trim() !== '';
  const depthValid = Number.isFinite(depthCm) && depthCm >= 0 && depthCm <= MAX_DEPTH_CM;

  const handleSave = () => {
    setSubmitted(true);
    if (unanswered.length > 0) {
      const offset = questionOffsets.current[unanswered[0].id];
      if (offset !== undefined) scrollRef.current?.scrollTo({ y: Math.max(0, offset - spacing.md), animated: true });
      return;
    }
    if (depthGiven && !depthValid) {
      if (depthOffset.current !== undefined) scrollRef.current?.scrollTo({ y: Math.max(0, depthOffset.current - spacing.md), animated: true });
      return;
    }

    // Built-in questions fill their named fields. One that wasn't asked (hidden, or follow-up only) is left
    // empty rather than keeping a value copied from the last visit.
    const field = (key: string) => fieldAnswer(asked, answers, key);
    const pain = field('pain');
    const trend = field('woundAppearanceTrend');
    updateTreatment(treatmentId, {
      assessment: {
        woundType: textAnswer(field('woundType')),
        exudateLevel: textAnswer(field('exudateLevel')),
        exudateType: textAnswer(field('exudateType')),
        infectionSigns: listAnswer(field('infectionSigns')),
        pain: typeof pain === 'number' ? pain : undefined,
        edgeCondition: textAnswer(field('edgeCondition')),
        periwoundCondition: textAnswer(field('periwoundCondition')),
        comorbidities: listAnswer(field('comorbidities')),
        woundBedTissue: listAnswer(field('woundBedTissue')),
        pressureStage: textAnswer(field('pressureStage')),
        burnDepth: textAnswer(field('burnDepth')),
        wagnerGrade: textAnswer(field('wagnerGrade')),
        ...(depthGiven ? { depthCm } : {}),
        ...(isTrend(trend) ? { woundAppearanceTrend: trend } : {}),
        responses: responsesFrom(asked, answers),
      },
    });

    completePrePhase(treatmentId);
    navigation.navigate('Camera', { treatmentId, step: 'post' });
  };

  return (
    <FormLayout
      mode="step"
      title="Clinical assessment"
      subtitle={
        <>
          {patientName && (
            <>
              <Text style={styles.patientName}>{patientName}</Text>,{' '}
            </>
          )}
          {visit}
        </>
      }
      accessory={
        <View style={styles.phaseTag}>
          <Text style={styles.phaseTagText}>{PHASE_LABEL[treatment.phase]}</Text>
        </View>
      }
      onClose={() => navigation.goBack()}
      submitLabel="Save and continue"
      onSubmit={handleSave}
      scrollRef={scrollRef}
    >
      {asked.map((question, i) => {
        const value = answerFor(question, answers);
        return (
          <View
            key={question.id}
            style={[styles.question, i > 0 && styles.questionDivider]}
            onLayout={(event) => {
              questionOffsets.current[question.id] = event.nativeEvent.layout.y;
            }}
          >
            <QuestionField
              question={question}
              value={value}
              onChange={(next) => setAnswer(answerKey(question), next)}
              error={submitted && question.required && value === undefined ? 'Choose an option to continue.' : undefined}
            />
          </View>
        );
      })}
      <View
        style={[styles.question, asked.length > 0 && styles.questionDivider]}
        onLayout={(event) => {
          depthOffset.current = event.nativeEvent.layout.y;
        }}
      >
        <FormField
          label="Wound depth (cm)"
          hint="Probe the deepest point. A photo cannot show depth. Leave empty if not measured."
          error={submitted && depthGiven && !depthValid ? `Enter the depth in centimetres, between 0 and ${MAX_DEPTH_CM}, or leave it empty.` : undefined}
          style={{ marginTop: 0 }}
        >
          <TextField
            value={depth}
            onChangeText={setDepth}
            keyboardType="decimal-pad"
            inputMode="decimal"
            placeholder="e.g. 0.8"
            accessibilityLabel="Wound depth in centimetres"
            invalid={submitted && depthGiven && !depthValid}
          />
        </FormField>
      </View>
    </FormLayout>
  );
};

const styles = StyleSheet.create({
  patientName: {
    fontWeight: '600',
    color: colors.textPrimary,
  },
  phaseTag: {
    justifyContent: 'center',
    height: 28,
    paddingHorizontal: 10,
    borderRadius: radii.small,
    backgroundColor: colors.surfacePressed,
  },
  phaseTagText: {
    fontSize: 13,
    lineHeight: 18,
    fontWeight: '600',
    color: colors.textPrimary,
  },
  question: {
    paddingVertical: 20,
  },
  questionDivider: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
  },
  notFound: {
    marginTop: spacing.lg,
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
