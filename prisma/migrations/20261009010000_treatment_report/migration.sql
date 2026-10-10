-- AlterTable
ALTER TABLE "Image" ADD COLUMN     "sha256" TEXT;

-- AlterTable
ALTER TABLE "AIResult" ADD COLUMN     "care" JSONB,
ADD COLUMN     "progress" JSONB,
ADD COLUMN     "rulesVersion" TEXT;
