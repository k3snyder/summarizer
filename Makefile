SHELL := /bin/bash

RUN_DIR := .run
RUN_DIR_ABS := $(CURDIR)/$(RUN_DIR)
BACKEND_PID_FILE := $(RUN_DIR_ABS)/backend.pid
FRONTEND_PID_FILE := $(RUN_DIR_ABS)/frontend.pid
BACKEND_LOG_FILE := $(RUN_DIR_ABS)/backend.log
FRONTEND_LOG_FILE := $(RUN_DIR_ABS)/frontend.log

BACKEND_DIR := backend
FRONTEND_DIR := frontend
BACKEND_HOST := 127.0.0.1
BACKEND_PORT := 8000
FRONTEND_PORT := 3000
BACKEND_PYTHON := $(BACKEND_DIR)/.venv/bin/python

.PHONY: services-up services-down services-status

services-up:
	@mkdir -p $(RUN_DIR)
	@if [ -f "$(BACKEND_PID_FILE)" ] && kill -0 "$$(cat "$(BACKEND_PID_FILE)")" 2>/dev/null; then \
		echo "Backend already running (pid $$(cat "$(BACKEND_PID_FILE)"))"; \
	else \
		if [ ! -x "$(BACKEND_PYTHON)" ]; then \
			echo "Missing $(BACKEND_PYTHON). Create the backend virtualenv first."; \
			exit 1; \
		fi; \
		echo "Starting backend on http://$(BACKEND_HOST):$(BACKEND_PORT)"; \
		nohup bash -lc 'cd "$(CURDIR)/$(BACKEND_DIR)" && exec "$(CURDIR)/$(BACKEND_PYTHON)" -m uvicorn app.main:app --reload --host $(BACKEND_HOST) --port $(BACKEND_PORT)' > "$(BACKEND_LOG_FILE)" 2>&1 & echo $$! > "$(BACKEND_PID_FILE)"; \
		sleep 2; \
		if ! kill -0 "$$(cat "$(BACKEND_PID_FILE)")" 2>/dev/null; then \
			echo "Backend failed to start. See $(BACKEND_LOG_FILE)"; \
			rm -f "$(BACKEND_PID_FILE)"; \
			exit 1; \
		fi; \
	fi
	@if [ -f "$(FRONTEND_PID_FILE)" ] && kill -0 "$$(cat "$(FRONTEND_PID_FILE)")" 2>/dev/null; then \
		echo "Frontend already running (pid $$(cat "$(FRONTEND_PID_FILE)"))"; \
	else \
		echo "Starting frontend on http://127.0.0.1:$(FRONTEND_PORT)"; \
		nohup bash -lc 'cd "$(CURDIR)/$(FRONTEND_DIR)" && exec npm run dev' > "$(FRONTEND_LOG_FILE)" 2>&1 & echo $$! > "$(FRONTEND_PID_FILE)"; \
		sleep 2; \
		if ! kill -0 "$$(cat "$(FRONTEND_PID_FILE)")" 2>/dev/null; then \
			echo "Frontend failed to start. See $(FRONTEND_LOG_FILE)"; \
			rm -f "$(FRONTEND_PID_FILE)"; \
			exit 1; \
		fi; \
	fi
	@$(MAKE) services-status

services-down:
	@if [ -f "$(FRONTEND_PID_FILE)" ]; then \
		pid="$$(cat "$(FRONTEND_PID_FILE)")"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			echo "Stopping frontend (pid $$pid)"; \
			kill "$$pid" 2>/dev/null || true; \
			sleep 1; \
			if kill -0 "$$pid" 2>/dev/null; then kill -9 "$$pid" 2>/dev/null || true; fi; \
		else \
			echo "Frontend pid file stale (pid $$pid)"; \
		fi; \
		rm -f "$(FRONTEND_PID_FILE)"; \
	else \
		echo "Frontend not running"; \
	fi
	@if [ -f "$(BACKEND_PID_FILE)" ]; then \
		pid="$$(cat "$(BACKEND_PID_FILE)")"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			echo "Stopping backend (pid $$pid)"; \
			kill "$$pid" 2>/dev/null || true; \
			sleep 1; \
			if kill -0 "$$pid" 2>/dev/null; then kill -9 "$$pid" 2>/dev/null || true; fi; \
		else \
			echo "Backend pid file stale (pid $$pid)"; \
		fi; \
		rm -f "$(BACKEND_PID_FILE)"; \
	else \
		echo "Backend not running"; \
	fi

services-status:
	@backend_status="stopped"; \
	if [ -f "$(BACKEND_PID_FILE)" ] && kill -0 "$$(cat "$(BACKEND_PID_FILE)")" 2>/dev/null; then \
		backend_status="running (pid $$(cat "$(BACKEND_PID_FILE)"))"; \
	fi; \
	frontend_status="stopped"; \
	if [ -f "$(FRONTEND_PID_FILE)" ] && kill -0 "$$(cat "$(FRONTEND_PID_FILE)")" 2>/dev/null; then \
		frontend_status="running (pid $$(cat "$(FRONTEND_PID_FILE)"))"; \
	fi; \
	echo "Backend: $$backend_status"; \
	echo "Frontend: $$frontend_status"; \
	echo "Backend log: $(BACKEND_LOG_FILE)"; \
	echo "Frontend log: $(FRONTEND_LOG_FILE)"
