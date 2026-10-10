import { Module } from '@nestjs/common';
import { APP_GUARD, APP_INTERCEPTOR } from '@nestjs/core';
import { LoggerModule } from 'nestjs-pino';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { AuthModule } from './auth/auth.module';
import { ClinicGuard } from './auth/clinic.guard';
import { SupabaseAuthGuard } from './auth/supabase-auth.guard';
import { CasesController } from './cases/cases.controller';
import { CasesService } from './cases/cases.service';
import { SummaryService } from './cases/summary.service';
import { ClinicController } from './clinic/clinic.controller';
import { ClinicService } from './clinic/clinic.service';
import { DashboardController } from './dashboard/dashboard.controller';
import { DashboardService } from './dashboard/dashboard.service';
import { ExportsController } from './exports/exports.controller';
import { ExportsService } from './exports/exports.service';
import { HealthController } from './health/health.controller';
import { PatientsController } from './patients/patients.controller';
import { PatientsService } from './patients/patients.service';
import { AuditService } from './platform/audit.service';
import { CacheService } from './platform/cache.service';
import { HttpCacheInterceptor } from './platform/http-cache.interceptor';
import { IdempotencyInterceptor } from './platform/idempotency.interceptor';
import { JobsService } from './platform/jobs.service';
import { RateLimitGuard } from './platform/rate-limit.guard';
import { RedisService } from './platform/redis.service';
import { SupabaseAdminService } from './platform/supabase-admin.service';
import { PrismaService } from './prisma.service';
import { QuestionsController } from './questions/questions.controller';
import { QuestionsService } from './questions/questions.service';
import { SharePublicController } from './share/share-public.controller';
import { ShareService } from './share/share.service';
import { UsersService } from './users.service';
import { ModelClient } from './visits/model-client';
import { StorageService } from './visits/storage.service';
import { SyncController } from './sync/sync.controller';
import { SyncService } from './sync/sync.service';
import { VisitsController } from './visits/visits.controller';
import { TreatmentReportService } from './visits/treatment-report.service';
import { VisitsService } from './visits/visits.service';

@Module({
  imports: [
    AuthModule,
    // One JSON log line per request, with ids and timings only: no bodies, tokens or patient data.
    LoggerModule.forRoot({
      pinoHttp: {
        level: process.env['LOG_LEVEL'] ?? 'info',
        redact: ['req.headers.authorization', 'req.headers.cookie', 'req.headers["idempotency-key"]'],
        serializers: { req: (req: { method: string; url: string; id: unknown }) => ({ id: req.id, method: req.method, url: req.url.split('?')[0] }) },
        autoLogging: { ignore: (req) => (req.url ?? '').startsWith('/api/health') },
      },
    }),
  ],
  controllers: [
    AppController,
    HealthController,
    ClinicController,
    PatientsController,
    CasesController,
    VisitsController,
    SyncController,
    DashboardController,
    ExportsController,
    SharePublicController,
    QuestionsController,
  ],
  providers: [
    AppService,
    PrismaService,
    RedisService,
    CacheService,
    AuditService,
    JobsService,
    SupabaseAdminService,
    StorageService,
    ModelClient,
    UsersService,
    SummaryService,
    PatientsService,
    CasesService,
    VisitsService,
    TreatmentReportService,
    SyncService,
    DashboardService,
    ClinicService,
    ExportsService,
    ShareService,
    QuestionsService,
    // Guards run in this order: valid sign-in → clinic membership and role → rate limit.
    { provide: APP_GUARD, useClass: SupabaseAuthGuard },
    { provide: APP_GUARD, useClass: ClinicGuard },
    { provide: APP_GUARD, useClass: RateLimitGuard },
    { provide: APP_INTERCEPTOR, useClass: HttpCacheInterceptor },
    { provide: APP_INTERCEPTOR, useClass: IdempotencyInterceptor },
  ],
})
export class AppModule {}
