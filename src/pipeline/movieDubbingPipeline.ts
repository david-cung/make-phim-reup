import type {
  DirtyFlags,
  JobSnapshot,
  Project,
  ProjectMediaState,
  SubtitleDoc,
} from "@/ipc/types";

export type MovieDubbingStage =
  | "mediaPreparation"
  | "transcription"
  | "translation"
  | "dubbingTextPreparation"
  | "voiceAssignment"
  | "ttsGeneration"
  | "durationMatching"
  | "audioTimelineAssembly"
  | "audioMixing"
  | "subtitlePreparation"
  | "export";

export type PipelineStageStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "invalid";

export interface PipelineStageState {
  stage: MovieDubbingStage;
  label: string;
  status: PipelineStageStatus;
  completedAt: string | null;
  inputVersion: string | null;
  outputVersion: number | null;
  progress: number | null;
  jobId: string | null;
  detail: string | null;
}

export interface MovieDubbingPipelineState {
  mediaPrepared: boolean;
  transcriptionCompleted: boolean;
  translationCompleted: boolean;
  dubbingPrepared: boolean;
  voicesGenerated: boolean;
  audioTimelineBuilt: boolean;
  audioMixed: boolean;
  subtitlesPrepared: boolean;
  exportCompleted: boolean;
  stages: PipelineStageState[];
}

type PipelineJobStage = JobSnapshot["stage"];

const STAGE_LABELS: Record<MovieDubbingStage, string> = {
  mediaPreparation: "Analyzing Movie",
  transcription: "Transcribing + Speakers",
  translation: "Translation + Validation",
  dubbingTextPreparation: "Dubbing Script",
  voiceAssignment: "Automatic Voices",
  ttsGeneration: "Generating Dubbing",
  durationMatching: "Synchronizing Audio",
  audioTimelineAssembly: "Audio Timeline",
  audioMixing: "Mixing",
  subtitlePreparation: "Subtitle Output",
  export: "Final Movie",
};

const STAGE_JOBS: Partial<Record<MovieDubbingStage, PipelineJobStage>> = {
  mediaPreparation: "extract_audio",
  transcription: "transcribe",
  translation: "translate",
  ttsGeneration: "tts",
  durationMatching: "sync",
  audioTimelineAssembly: "sync",
  audioMixing: "mix",
  export: "render",
};

export const PIPELINE_INVALIDATION_RULES: Record<string, MovieDubbingStage[]> = {
  transcriptionChanged: [
    "translation",
    "dubbingTextPreparation",
    "ttsGeneration",
    "durationMatching",
    "audioTimelineAssembly",
    "audioMixing",
    "export",
  ],
  translationChanged: [
    "dubbingTextPreparation",
    "ttsGeneration",
    "durationMatching",
    "audioTimelineAssembly",
    "audioMixing",
    "export",
  ],
  dubbingTextChanged: [
    "ttsGeneration",
    "durationMatching",
    "audioTimelineAssembly",
    "audioMixing",
    "export",
  ],
  voiceChanged: [
    "ttsGeneration",
    "durationMatching",
    "audioTimelineAssembly",
    "audioMixing",
    "export",
  ],
  subtitleStyleChanged: ["subtitlePreparation", "export"],
};

function dirtyInvalidates(
  dirty: DirtyFlags | null | undefined,
  stage: MovieDubbingStage,
): boolean {
  if (!dirty) return false;
  if (stage === "ttsGeneration") return dirty.tts;
  if (stage === "durationMatching" || stage === "audioTimelineAssembly") {
    return dirty.sync;
  }
  if (stage === "audioMixing") return dirty.mix;
  if (stage === "export") return dirty.render;
  return false;
}

function newestJob(
  jobs: JobSnapshot[],
  stage: PipelineJobStage | undefined,
): JobSnapshot | null {
  if (!stage) return null;
  const staged = jobs
    .filter((job) => job.stage === stage)
    .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
  return staged[0] ?? null;
}

function jobStatus(job: JobSnapshot | null): PipelineStageStatus | null {
  if (!job) return null;
  if (job.status === "queued" || job.status === "running") return "running";
  if (job.status === "failed") return "failed";
  if (job.status === "cancelled") return "cancelled";
  return null;
}

function stage(
  stageName: MovieDubbingStage,
  args: {
    done: boolean;
    invalid?: boolean;
    completedAt?: string | null;
    inputVersion?: string | null;
    outputVersion?: number | null;
    detail?: string | null;
    jobs: JobSnapshot[];
  },
): PipelineStageState {
  const job = newestJob(args.jobs, STAGE_JOBS[stageName]);
  const liveStatus = jobStatus(job);
  return {
    stage: stageName,
    label: STAGE_LABELS[stageName],
    status:
      liveStatus ??
      (args.invalid ? "invalid" : args.done ? "completed" : "pending"),
    completedAt: args.completedAt ?? null,
    inputVersion: args.inputVersion ?? null,
    outputVersion: args.outputVersion ?? null,
    progress: job && liveStatus === "running" ? job.progress : null,
    jobId: job && liveStatus === "running" ? job.id : null,
    detail:
      liveStatus === "failed"
        ? job?.errorMessage ?? args.detail ?? null
        : args.detail ?? null,
  };
}

export function computeMovieDubbingPipeline(input: {
  project: Project | null;
  media: ProjectMediaState | null;
  subtitleDoc: SubtitleDoc | null;
  jobsById: Record<string, JobSnapshot>;
}): MovieDubbingPipelineState {
  const projectId = input.project?.id ?? null;
  const media = input.media;
  const subtitleDoc = input.subtitleDoc;
  const projectJobs = Object.values(input.jobsById).filter(
    (job) => !!projectId && job.projectId === projectId,
  );
  const dirty = subtitleDoc?.dirty ?? media?.subtitles?.dirty ?? null;
  const segmentCount =
    subtitleDoc?.segments.length ?? media?.subtitles?.segmentCount ?? 0;
  const translatedCount = media?.translation?.translatedCount ?? 0;
  const translationCount = media?.translation?.segmentCount ?? 0;
  const dubbingPrepared =
    !!subtitleDoc &&
    subtitleDoc.segments.length > 0 &&
    subtitleDoc.segments.every((segment) =>
      Boolean(
        (segment.dubbingText || segment.translatedText || segment.sourceText).trim(),
      ),
    );
  const voiceAssignment =
    !!subtitleDoc &&
    subtitleDoc.segments.length > 0 &&
    subtitleDoc.segments.every((segment) =>
      Boolean(segment.voiceId || input.project?.ttsVoiceId),
    );
  const ttsReady =
    !!media?.tts &&
    media.tts.subtitleCount > 0 &&
    media.tts.generatedCount >= media.tts.subtitleCount &&
    media.tts.missingCount === 0 &&
    media.tts.staleCount === 0 &&
    !dirty?.tts;
  const syncReady =
    !!media?.sync &&
    media.sync.subtitleCount > 0 &&
    media.sync.syncedCount >= media.sync.subtitleCount &&
    media.sync.missingCount === 0 &&
    media.sync.staleCount === 0 &&
    !dirty?.sync;
  const mixReady = media?.mix?.status === "ready" && !dirty?.mix;
  const renderReady = media?.render?.status === "ready" && !dirty?.render;
  const subtitlesReady = !!media?.subtitles && !dirty?.render;

  return {
    mediaPrepared: !!media?.audioAbsolutePath,
    transcriptionCompleted: !!media?.transcript,
    translationCompleted:
      !!media?.translation &&
      translationCount > 0 &&
      translatedCount >= translationCount,
    dubbingPrepared,
    voicesGenerated: ttsReady,
    audioTimelineBuilt: syncReady,
    audioMixed: mixReady,
    subtitlesPrepared: subtitlesReady,
    exportCompleted: renderReady,
    stages: [
      stage("mediaPreparation", {
        done: !!media?.audioAbsolutePath,
        completedAt: media?.audio?.createdAt ?? null,
        inputVersion:
          input.project?.sourceHash ?? input.project?.sourceMediaPath ?? null,
        outputVersion: media?.audio ? 1 : null,
        detail: media?.audioAbsolutePath ? "Analyzing movie complete" : null,
        jobs: projectJobs,
      }),
      stage("transcription", {
        done: !!media?.transcript,
        completedAt: media?.transcript?.createdAt ?? null,
        inputVersion: media?.transcript?.audioHash ?? null,
        outputVersion: media?.transcript ? 1 : null,
        detail: media?.transcript
          ? `${media.transcript.segmentCount} dialogue segments`
          : null,
        jobs: projectJobs,
      }),
      stage("translation", {
        done:
          !!media?.translation &&
          translationCount > 0 &&
          translatedCount >= translationCount,
        completedAt: media?.translation?.updatedAt ?? null,
        inputVersion: media?.translation?.transcriptCacheKey ?? null,
        outputVersion: media?.translation ? 1 : null,
        detail: media?.translation
          ? `${translatedCount}/${translationCount} translated and validated`
          : null,
        jobs: projectJobs,
      }),
      stage("dubbingTextPreparation", {
        done: dubbingPrepared,
        invalid: dirtyInvalidates(dirty, "ttsGeneration"),
        completedAt:
          subtitleDoc?.updatedAt ?? media?.subtitles?.updatedAt ?? null,
        inputVersion:
          media?.translation?.cacheKey ??
          media?.subtitles?.derivedFrom.translationCacheKey ??
          null,
        outputVersion: subtitleDoc?.version ?? null,
        detail: segmentCount ? `${segmentCount} dubbing lines prepared` : null,
        jobs: projectJobs,
      }),
      stage("voiceAssignment", {
        done: voiceAssignment,
        invalid: dirtyInvalidates(dirty, "ttsGeneration"),
        completedAt:
          subtitleDoc?.updatedAt ?? media?.subtitles?.updatedAt ?? null,
        inputVersion: input.project?.ttsVoiceId ?? null,
        outputVersion: subtitleDoc?.version ?? null,
        detail: voiceAssignment ? "Movie-scoped voice mapping ready" : null,
        jobs: projectJobs,
      }),
      stage("ttsGeneration", {
        done: ttsReady,
        invalid: dirtyInvalidates(dirty, "ttsGeneration"),
        completedAt: media?.tts?.updatedAt ?? null,
        inputVersion: media?.tts
          ? `${media.tts.engine}:${media.tts.defaultVoiceId}`
          : null,
        outputVersion: media?.tts ? 1 : null,
        detail: media?.tts
          ? `${media.tts.generatedCount}/${media.tts.subtitleCount} dubbed`
          : null,
        jobs: projectJobs,
      }),
      stage("durationMatching", {
        done: syncReady,
        invalid: dirtyInvalidates(dirty, "durationMatching"),
        completedAt: media?.sync?.updatedAt ?? null,
        inputVersion: media?.tts?.updatedAt ?? null,
        outputVersion: media?.sync ? 1 : null,
        detail: media?.sync
          ? `${media.sync.syncedCount}/${media.sync.subtitleCount} synced, ${media.sync.tooLongCount} too long`
          : null,
        jobs: projectJobs,
      }),
      stage("audioTimelineAssembly", {
        done: syncReady,
        invalid: dirtyInvalidates(dirty, "audioTimelineAssembly"),
        completedAt: media?.sync?.updatedAt ?? null,
        inputVersion: media?.sync?.updatedAt ?? null,
        outputVersion: media?.sync ? 1 : null,
        detail: syncReady ? "Vietnamese speech aligned to dialogue timing" : null,
        jobs: projectJobs,
      }),
      stage("audioMixing", {
        done: mixReady,
        invalid:
          dirtyInvalidates(dirty, "audioMixing") ||
          media?.mix?.status === "stale",
        completedAt: media?.mix?.generatedAt ?? null,
        inputVersion: media?.mix?.relativePath ?? null,
        outputVersion: media?.mix ? 1 : null,
        detail: media?.mix?.warning ?? media?.mix?.relativePath ?? null,
        jobs: projectJobs,
      }),
      stage("subtitlePreparation", {
        done: subtitlesReady,
        invalid: dirtyInvalidates(dirty, "subtitlePreparation"),
        completedAt: media?.subtitles?.updatedAt ?? null,
        inputVersion: media?.subtitles?.derivedFrom.translationCacheKey ?? null,
        outputVersion: subtitleDoc?.version ?? null,
        detail: media?.subtitles ? `${media.subtitles.segmentCount} subtitle lines` : null,
        jobs: projectJobs,
      }),
      stage("export", {
        done: renderReady,
        invalid:
          dirtyInvalidates(dirty, "export") ||
          media?.render?.status === "stale",
        completedAt: media?.render?.generatedAt ?? null,
        inputVersion: media?.render?.relativePath ?? null,
        outputVersion: media?.render ? 1 : null,
        detail: media?.render?.warning ?? media?.render?.relativePath ?? null,
        jobs: projectJobs,
      }),
    ],
  };
}

export function stageStatusForRail(
  status: PipelineStageStatus,
): "waiting" | "active" | "done" | "error" {
  if (status === "running") return "active";
  if (status === "completed") return "done";
  if (status === "failed" || status === "cancelled" || status === "invalid") {
    return "error";
  }
  return "waiting";
}
