import { Controller, Get, Param, Query, Req, Res } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { Response } from 'express';
import { Roles, clinicCtx, type ClinicRequest } from '../auth/clinic.guard';
import { Limit } from '../platform/rate-limit.guard';
import { ExportsService } from './exports.service';

@ApiTags('exports')
@ApiBearerAuth()
@Roles('ADMIN')
@Controller('exports')
export class ExportsController {
  constructor(private readonly exports: ExportsService) {}

  @Get(':dataset')
  @Limit({ max: 10, windowSeconds: 3600, bucket: 'export' })
  @ApiOperation({ summary: 'CSV of patients, visits or validation (nurse assessment next to model findings), streamed. De-identified unless deidentify=false (validation: always). Audited first.' })
  async download(@Param('dataset') dataset: string, @Query('deidentify') deidentify: string | undefined, @Req() req: ClinicRequest, @Res() res: Response) {
    const deid = deidentify !== 'false';
    const rows = this.exports.stream(clinicCtx(req), dataset.replace(/\.csv$/, ''), deid);
    const first = await rows.next(); // validates and writes the audit entry before any header is sent
    const date = new Date().toISOString().slice(0, 10);
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="wound-${dataset.replace(/\.csv$/, '')}-${date}${deid ? '-deid' : ''}.csv"`);
    res.setHeader('Cache-Control', 'no-store');
    if (!first.done) res.write(first.value);
    for await (const chunk of rows) {
      if (!res.write(chunk)) await new Promise((resolve) => res.once('drain', resolve));
    }
    res.end();
  }
}
