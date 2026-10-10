-- AlterTable
ALTER TABLE "ClinicalAssessment" ADD COLUMN     "burnDepth" TEXT,
ADD COLUMN     "pressureStage" TEXT,
ADD COLUMN     "wagnerGrade" TEXT,
ADD COLUMN     "woundBedTissue" TEXT[] DEFAULT ARRAY[]::TEXT[];
