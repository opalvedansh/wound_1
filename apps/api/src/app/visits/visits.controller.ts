import { Body, Controller, Delete, Get, HttpCode, Param, Post, Req, UploadedFile, UseInterceptors } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ApiBearerAuth, ApiConsumes, ApiOperation, ApiTags } from '@nestjs/swagger';
import { Roles, clinicCtx, type ClinicRequest } from '../auth/clinic.guard';
import { Limit } from '../platform/rate-limit.guard';
import { MAX_PHOTO_BYTES, VisitsService, type PhotoUpload } from './visits.service';

/** Photo → background analysis → clinician review. Clinical data: doctors and admins only. */
@ApiTags('visits')
@ApiBearerAuth()
@Roles('ADMIN', 'DOCTOR')
@Controller()
export class VisitsController {
  constructor(private readonly visits: VisitsService) {}

  @Get('model/intake-questions')
  intakeQuestions() {
    return this.visits.intakeQuestions();
  }

  @Post('model/follow-ups')
  followUps(@Body() body: unknown) {
    return this.visits.followUps(body);
  }

  @Post('model/photo-check')
  @Limit({ max: 30, windowSeconds: 60, bucket: 'upload' })
  @ApiConsumes('multipart/form-data')
  @ApiOperation({ summary: "A photo's quality and whether the calibration sticker is in it, before it is used. Nothing is stored." })
  @UseInterceptors(FileInterceptor('photo', { limits: { fileSize: MAX_PHOTO_BYTES, files: 1 } }))
  photoCheck(@UploadedFile() photo: PhotoUpload | undefined) {
    return this.visits.photoCheck(photo);
  }

  @Post('cases/:caseId/visits')
  @Limit({ max: 30, windowSeconds: 60, bucket: 'upload' })
  @ApiConsumes('multipart/form-data')
  @ApiOperation({ summary: 'Upload a wound photo with intake answers. Returns at once with status "processing"; poll GET /visits/:id.' })
  @UseInterceptors(FileInterceptor('photo', { limits: { fileSize: MAX_PHOTO_BYTES, files: 1 } }))
  create(@Param('caseId') caseId: string, @UploadedFile() photo: PhotoUpload | undefined, @Body('intake') intake: unknown, @Req() req: ClinicRequest) {
    return this.visits.create(clinicCtx(req), caseId, photo, intake);
  }

  @Get('visits/:id')
  get(@Param('id') id: string, @Req() req: ClinicRequest) {
    return this.visits.get(clinicCtx(req), id);
  }

  @Post('visits/:id/retry')
  retry(@Param('id') id: string, @Req() req: ClinicRequest) {
    return this.visits.retry(clinicCtx(req), id);
  }

  @Post('visits/:id/review')
  @ApiOperation({ summary: "Approve, edit or reject a visit's draft report. One review per visit." })
  review(@Param('id') id: string, @Body() body: unknown, @Req() req: ClinicRequest) {
    return this.visits.review(clinicCtx(req), id, body);
  }

  @Delete('visits/:id')
  @HttpCode(204)
  remove(@Param('id') id: string, @Req() req: ClinicRequest) {
    return this.visits.remove(clinicCtx(req), id);
  }
}
