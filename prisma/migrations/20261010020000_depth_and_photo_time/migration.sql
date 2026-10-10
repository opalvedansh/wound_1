-- AlterTable
ALTER TABLE "ClinicalAssessment" ADD COLUMN     "depthCm" DOUBLE PRECISION;

-- AlterTable
ALTER TABLE "Image" ADD COLUMN     "takenAt" TIMESTAMP(3);
