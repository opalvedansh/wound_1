import { Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import { Queue, Worker, type JobsOptions } from 'bullmq';

export type JobName = 'analyze-visit' | 'thumbnail' | 'forward-review' | 'purge' | 'treatment-report';
/** `final` is true on the last attempt, so a handler can record a permanent failure instead of retrying. */
export interface JobMeta {
  final: boolean;
}
type Handler = (data: Record<string, unknown>, meta: JobMeta) => Promise<void>;

const QUEUE = 'wound';
const DEFAULTS: JobsOptions = {
  attempts: 3,
  backoff: { type: 'exponential', delay: 10_000 },
  removeOnComplete: { age: 3600, count: 1000 },
  removeOnFail: { age: 7 * 24 * 3600 },
};

/**
 * Background work (BullMQ on Redis): model analysis, thumbnails, review forwarding, purges. Any API instance
 * enqueues; workers run in processes started with RUN_WORKERS=1 (the default for a single instance; set it to 0
 * on web instances and run a separate worker service to scale). Without Redis, a job runs in this process right
 * after the request instead, so development works with no extra services.
 */
@Injectable()
export class JobsService implements OnModuleDestroy {
  private readonly logger = new Logger(JobsService.name);
  private readonly handlers = new Map<JobName, Handler>();
  private queue?: Queue | null;
  private worker?: Worker;

  private connection() {
    const url = process.env['REDIS_URL'];
    if (!url) return null;
    const u = new URL(url);
    return {
      host: u.hostname,
      port: Number(u.port || 6379),
      username: u.username || undefined,
      password: u.password ? decodeURIComponent(u.password) : undefined,
      tls: u.protocol === 'rediss:' ? {} : undefined,
      maxRetriesPerRequest: null, // required by BullMQ workers
    };
  }

  private getQueue(): Queue | null {
    if (this.queue === undefined) {
      const connection = this.connection();
      this.queue = connection ? new Queue(QUEUE, { connection, defaultJobOptions: DEFAULTS }) : null;
    }
    return this.queue;
  }

  /** Registers the function that runs a job; called by the services that own each job at startup. */
  handle(name: JobName, handler: Handler) {
    this.handlers.set(name, handler);
  }

  async enqueue(name: JobName, data: Record<string, unknown>, options: JobsOptions = {}): Promise<void> {
    const queue = this.getQueue();
    if (queue) {
      try {
        await queue.add(name, data, options);
        return;
      } catch (error) {
        this.logger.warn(`Queue unavailable, running ${name} here: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    setImmediate(() => void this.run(name, data, { final: true }).catch(() => undefined));
  }

  /** Starts processing jobs in this process (RUN_WORKERS=1). */
  startWorkers(concurrency = Number(process.env['WORKER_CONCURRENCY'] ?? 4)) {
    const connection = this.connection();
    if (!connection || this.worker) return;
    this.worker = new Worker(
      QUEUE,
      (job) =>
        this.run(job.name as JobName, job.data as Record<string, unknown>, {
          final: job.attemptsMade + 1 >= (job.opts.attempts ?? 1),
        }),
      {
        connection,
        concurrency,
        // While the queue is empty the worker waits on Redis this long per call (a new job wakes it at once).
        // Longer waits mean far fewer Redis commands when idle, which matters on a metered Redis.
        drainDelay: Number(process.env['QUEUE_IDLE_SECONDS'] ?? 30),
        stalledInterval: 120_000,
      },
    );
    this.worker.on('failed', (job, error) => this.logger.warn(`Job ${job?.name} ${job?.id} failed: ${error.message}`));
    // Daily clean-up of retakes, failed visits and old tombstones.
    void this.getQueue()?.upsertJobScheduler('purge-daily', { pattern: '30 3 * * *' }, { name: 'purge', data: {} });
    this.logger.log(`Background workers started (concurrency ${concurrency}).`);
  }

  private async run(name: JobName, data: Record<string, unknown>, meta: JobMeta) {
    const handler = this.handlers.get(name);
    if (!handler) throw new Error(`No handler for job ${name}`);
    try {
      await handler(data, meta);
    } catch (error) {
      this.logger.warn(`${name} failed: ${error instanceof Error ? error.message : String(error)}`);
      throw error;
    }
  }

  async onModuleDestroy() {
    await this.worker?.close();
    await this.queue?.close();
  }
}
