"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { DocumentViewer } from "@/components/output/document-viewer";
import { PageCard } from "@/components/output/page-card";
import { api, ApiError } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import type { DocumentOutput } from "@/types";

type PageState =
  | { status: "loading" }
  | { status: "success"; data: DocumentOutput }
  | { status: "not_found" }
  | { status: "not_completed"; message: string }
  | { status: "error"; message: string };

function LoadingSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="space-y-3 flex-1">
              <div className="h-6 w-48 bg-muted rounded" />
              <div className="h-4 w-64 bg-muted/60 rounded" />
            </div>
            <div className="h-9 w-28 bg-muted rounded" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 mb-6">
            <div className="h-4 w-32 bg-muted/60 rounded" />
            <div className="h-4 w-32 bg-muted/60 rounded" />
          </div>
          <div className="h-10 w-full bg-muted rounded mb-6" />
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-24 bg-muted/40 rounded-lg" />
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function ErrorDisplay({
  title,
  message,
  showHomeLink = true,
}: {
  title: string;
  message: string;
  showHomeLink?: boolean;
}) {
  return (
    <Card className="border-destructive/50 bg-destructive/5">
      <CardContent className="p-8">
        <div className="flex flex-col items-center text-center gap-4">
          <div className="w-16 h-16 rounded-full bg-destructive/10 flex items-center justify-center">
            <svg
              className="w-8 h-8 text-destructive"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-destructive">{title}</h2>
            <p className="text-muted-foreground mt-2">{message}</p>
          </div>
          {showHomeLink && (
            <Button asChild className="mt-4">
              <Link href="/">
                <svg
                  className="w-4 h-4 mr-2"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M10 19l-7-7m0 0l7-7m-7 7h18"
                  />
                </svg>
                Back to Home
              </Link>
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function NotCompletedDisplay({ message }: { message: string }) {
  return (
    <Card className="border-amber-500/50 bg-amber-500/5">
      <CardContent className="p-8">
        <div className="flex flex-col items-center text-center gap-4">
          <div className="w-16 h-16 rounded-full bg-amber-500/10 flex items-center justify-center">
            <svg
              className="w-8 h-8 text-amber-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-amber-600">
              Job Not Completed
            </h2>
            <p className="text-muted-foreground mt-2">{message}</p>
          </div>
          <Button asChild className="mt-4">
            <Link href="/">
              <svg
                className="w-4 h-4 mr-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M10 19l-7-7m0 0l7-7m-7 7h18"
                />
              </svg>
              Back to Home
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function JobResultsPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;

  const [pageState, setPageState] = useState<PageState>({ status: "loading" });

  useEffect(() => {
    if (!jobId) return;

    async function fetchOutput() {
      try {
        const output = await api.getJobOutput(jobId);
        setPageState({ status: "success", data: output });
      } catch (err) {
        if (err instanceof ApiError) {
          if (err.status === 404) {
            setPageState({ status: "not_found" });
          } else if (err.status === 400) {
            setPageState({
              status: "not_completed",
              message:
                err.detail ??
                "This job has not completed processing yet. Please check back later.",
            });
          } else {
            setPageState({
              status: "error",
              message: err.detail ?? err.message,
            });
          }
        } else if (err instanceof Error) {
          setPageState({ status: "error", message: err.message });
        } else {
          setPageState({
            status: "error",
            message: "An unexpected error occurred while fetching job output.",
          });
        }
      }
    }

    fetchOutput();
  }, [jobId]);

  return (
    <div className="min-h-screen p-8">
      <main className="max-w-4xl mx-auto space-y-8">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" asChild>
            <Link href="/">
              <svg
                className="w-4 h-4 mr-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M10 19l-7-7m0 0l7-7m-7 7h18"
                />
              </svg>
              Back
            </Link>
          </Button>
          <div>
            <h1 className="text-2xl font-bold">Job Results</h1>
            <p className="text-sm text-muted-foreground font-mono">{jobId}</p>
          </div>
        </div>

        {pageState.status === "loading" && <LoadingSkeleton />}

        {pageState.status === "not_found" && (
          <ErrorDisplay
            title="Job Not Found"
            message="The requested job could not be found. It may have been deleted or the ID is incorrect."
          />
        )}

        {pageState.status === "not_completed" && (
          <NotCompletedDisplay message={pageState.message} />
        )}

        {pageState.status === "error" && (
          <ErrorDisplay title="Error Loading Results" message={pageState.message} />
        )}

        {pageState.status === "success" && (
          <DocumentViewer output={pageState.data}>
            <div className="space-y-4">
              {pageState.data.pages.map((page, index) => (
                <PageCard
                  key={page.chunk_id}
                  page={page}
                  pageNumber={index + 1}
                  defaultExpanded={index === 0}
                />
              ))}
            </div>
          </DocumentViewer>
        )}
      </main>
    </div>
  );
}
