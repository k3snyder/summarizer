"use client";

import * as React from "react";
import { useCallback, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const ACCEPTED_FILE_TYPES = [".pdf", ".pptx", ".txt", ".md"];
const ACCEPTED_MIME_TYPES = [
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "text/plain",
  "text/markdown",
  "text/x-markdown",
];
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onRemove: () => void;
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function getFileExtension(filename: string): string {
  const lastDot = filename.lastIndexOf(".");
  return lastDot !== -1 ? filename.slice(lastDot).toLowerCase() : "";
}

function isValidFileType(file: File): boolean {
  const extension = getFileExtension(file.name);
  return (
    ACCEPTED_FILE_TYPES.includes(extension) ||
    ACCEPTED_MIME_TYPES.includes(file.type)
  );
}

function isValidFileSize(file: File): boolean {
  return file.size <= MAX_FILE_SIZE;
}

function getFileIcon(filename: string): React.ReactNode {
  const extension = getFileExtension(filename);
  if (extension === ".pdf") {
    return (
      <svg
        className="w-12 h-12 text-red-400"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
        />
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9 13h6m-6 4h4"
        />
      </svg>
    );
  }
  if (extension === ".pptx") {
    return (
      <svg
        className="w-12 h-12 text-orange-400"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
        />
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9 11h2a2 2 0 110 4H9v2m0-6v6"
        />
      </svg>
    );
  }
  return (
    <svg
      className="w-12 h-12 text-blue-400"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
      />
    </svg>
  );
}

export function FileUpload({
  onFileSelect,
  selectedFile,
  onRemove,
}: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const validateAndSelectFile = useCallback(
    (file: File) => {
      setError(null);

      if (!isValidFileType(file)) {
        setError(
          `Invalid file type. Please upload a PDF, PowerPoint, TXT, or MD file.`
        );
        return;
      }

      if (!isValidFileSize(file)) {
        setError(`File is too large. Maximum size is 100MB.`);
        return;
      }

      onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        validateAndSelectFile(files[0]);
      }
    },
    [validateAndSelectFile]
  );

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        validateAndSelectFile(files[0]);
      }
    },
    [validateAndSelectFile]
  );

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleRemove = useCallback(() => {
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
    onRemove();
  }, [onRemove]);

  if (selectedFile) {
    return (
      <Card className="border-2 border-dashed border-primary/50 bg-card/50">
        <CardContent className="p-6">
          <div className="flex items-center gap-4">
            {getFileIcon(selectedFile.name)}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{selectedFile.name}</p>
              <p className="text-xs text-muted-foreground">
                {formatFileSize(selectedFile.size)}
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRemove}
              className="text-muted-foreground hover:text-destructive"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
              <span className="sr-only">Remove file</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div>
      <Card
        className={cn(
          "border-2 border-dashed cursor-pointer transition-colors",
          isDragOver
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50",
          error && "border-destructive"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <CardContent className="p-8">
          <div className="flex flex-col items-center justify-center text-center">
            <div className="p-3 rounded-full bg-muted mb-4">
              <svg
                className="w-8 h-8 text-muted-foreground"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>
            <p className="text-sm font-medium mb-1">
              Drag and drop your file here
            </p>
            <p className="text-xs text-muted-foreground mb-3">
              or click to browse
            </p>
            <p className="text-xs text-muted-foreground">
              PDF, PowerPoint, TXT, or MD files up to 100MB
            </p>
          </div>
        </CardContent>
      </Card>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_FILE_TYPES.join(",")}
        onChange={handleFileInputChange}
        className="hidden"
      />
      {error && (
        <p className="mt-2 text-sm text-destructive">{error}</p>
      )}
    </div>
  );
}
