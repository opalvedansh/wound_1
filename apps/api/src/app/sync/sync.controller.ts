import { Body, Controller, Get, Param, ParseUUIDPipe, Post, Query, Req, UploadedFile, UseInterceptors } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ApiBearerAuth, ApiConsumes, ApiOperation, ApiTags } from '@nestjs/swagger';
import { Roles, clinicCtx, type ClinicRequest } from '../auth/clinic.guard';
import { Limit } from '../platform/rate-limit.guard';
import { MAX_PHOTO_BYTES, type PhotoUpload } from '../visits/visits.service';
import { SyncService } from './sync.service';

/** The mobile app's offline-first sync. The front desk syncs patients only. */
@ApiTags('sync')
@ApiBearerAuth()
@Controller('sync')
export class SyncController {
  constructor(private readonly sync: SyncService) {}

  @Post('push')
  @Limit({ max: 120, windowSeconds: 60, bucket: 'sync' })
  @ApiOperation({ summary: 'Send local changes (changed fields per record, deletes). Bad records are listed in `rejected`; the rest are applied.' })
  push(@Body() body: unknown, @Req() req: ClinicRequest) {
    return this.sync.push(clinicCtx(req), body);
  }

  @Get('pull')
  @Limit({ max: 120, windowSeconds: 60, bucket: 'sync' })
  @ApiOperation({ summary: 'Records changed since `since` (the last serverTime). Repeat with `cursor` while it is not null.' })
  pull(@Query('since') since: unknown, @Query('cursor') cursor: unknown, @Req() req: ClinicRequest) {
    return this.sync.pull(clinicCtx(req), since, cursor);
  }

  @Post('treatments/:id/photos/:phase')
  @Roles('ADMIN', 'DOCTOR')
  @Limit({ max: 30, windowSeconds: 60, bucket: 'upload' })
  @ApiConsumes('multipart/form-data')
  @ApiOperation({
    summary:
      "Upload the pre- or post-treatment photo of a synced treatment, optionally with `measured_length_cm` (the wound's longest length by ruler, which gives a photo without a sticker its scale) and `taken_at` (when the phone took it, ISO). The pre photo is analysed in the background.",
  })
  @UseInterceptors(FileInterceptor('photo', { limits: { fileSize: MAX_PHOTO_BYTES, files: 1 } }))
  photo(
    @Param('id', new ParseUUIDPipe()) id: string,
    @Param('phase') phase: string,
    @UploadedFile() photo: PhotoUpload | undefined,
    @Body('measured_length_cm') measuredLengthCm: unknown,
    @Body('taken_at') takenAt: unknown,
    @Req() req: ClinicRequest,
  ) {
    return this.sync.photo(clinicCtx(req), id, phase, photo, measuredLengthCm, takenAt);
  }
}
