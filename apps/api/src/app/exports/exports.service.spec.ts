import type { AuditService } from '../platform/audit.service';
import type { PrismaService } from '../prisma.service';
import { ExportsService } from './exports.service';

const visit = (overrides: Record<string, unknown> = {}) => ({
  id: 'r1',
  createdAt: new Date(Date.UTC(2026, 9, 3)),
  area: 4.2,
  reviewStatus: 'edited',
  modelVersions: { wound_type: 'v1' },
  findings: {
    wound_type: { label: 'pressure', prob: 0.81 },
    severity: { pu_stage: { label: 'stage_3', prob: 0.66 } },
    tissue_pct: { granulation: 70, slough: 30 },
    tissue_untrusted: ['necrosis'],
    periwound_erythema_frac: 0.12,
    flags_version: 'flags-0.1-unsigned',
  },
  review: { corrections: { wound_type: 'pressure' } },
  phase: {
    assessment: {
      woundBedTissue: ['Granulation', 'Slough'],
      pressureStage: 'Stage 3',
      burnDepth: null,
      wagnerGrade: null,
      periwoundCondition: 'Erythematous',
      edgeCondition: 'Rolled',
      infectionSigns: ['Erythema', 'Malodor'],
      exudateLevel: 'Moderate',
    },
    treatment: {
      sequence: 2,
      deletedAt: null,
      case: { id: 'c1', location: 'sacrum', woundType: 'Pressure Ulcer', patient: { patientId: 'P-007' } },
    },
  },
  ...overrides,
});

function setup(rows: unknown[]) {
  const findMany = jest.fn().mockResolvedValueOnce(rows).mockResolvedValue([]);
  const prisma = { aIResult: { findMany } } as unknown as PrismaService;
  const audit = { logNow: jest.fn(async () => undefined) } as unknown as AuditService;
  return { service: new ExportsService(prisma, audit), audit, findMany };
}

const collect = async (gen: AsyncGenerator<string>) => {
  let out = '';
  for await (const chunk of gen) out += chunk;
  return out;
};

describe('ExportsService validation dataset', () => {
  it('puts the nurse assessment next to the model findings, de-identified even when asked not to be', async () => {
    const { service, audit } = setup([visit()]);
    const csv = await collect(service.stream({ clinicId: 'k1', userId: 'u1' } as never, 'validation', false));
    const [header, row] = csv.trim().split('\n');
    expect(header).not.toMatch(/name/);
    expect(header.split(',')).toEqual(expect.arrayContaining(['nurse_pressure_stage', 'model_severity', 'model_tissue_pct']));
    expect(row).toContain('P-007,c1,T2,2026-10,sacrum,Pressure Ulcer,Granulation;Slough,Stage 3,,,Erythematous,Rolled,Erythema;Malodor,Moderate,pressure,0.81');
    expect(row).toContain('stage_3');
    expect(row).toContain('flags-0.1-unsigned');
    expect(audit.logNow).toHaveBeenCalledWith(expect.objectContaining({ action: 'export.validation', details: { deidentified: true } }));
  });

  it('skips deleted treatments and visits without an assessment still export the model side', async () => {
    const deleted = visit({ id: 'r2', phase: { ...visit().phase, treatment: { ...visit().phase.treatment, deletedAt: new Date() } } });
    const noAssessment = visit({ id: 'r3', phase: { ...visit().phase, assessment: null } });
    const { service } = setup([deleted, noAssessment]);
    const rows = (await collect(service.stream({ clinicId: 'k1', userId: 'u1' } as never, 'validation', true))).trim().split('\n');
    expect(rows).toHaveLength(2);
    expect(rows[1]).toContain('P-007,c1,T2,2026-10,sacrum,Pressure Ulcer,,,,,,,,,pressure');
  });
});
