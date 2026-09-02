export function pageWindow(total, limit, offset) {
    const safeTotal = Math.max(0, Number(total) || 0);
    const safeLimit = Math.max(1, Number(limit) || 1);
    const safeOffset = Math.max(0, Number(offset) || 0);
    return {
        start: safeTotal ? Math.min(safeOffset + 1, safeTotal) : 0,
        end: Math.min(safeOffset + safeLimit, safeTotal),
        hasPrevious: safeOffset > 0,
        hasNext: safeOffset + safeLimit < safeTotal,
    };
}

export function readExactFilters(form) {
    const data = new FormData(form);
    const filters = {};
    for (const [key, rawValue] of data.entries()) {
        const value = String(rawValue).trim();
        if (value) filters[key] = value;
    }
    return filters;
}

export const SAVED_MEMORIES_EXPLANATION = (
    'Saved memories are global structured records and survive individual chat-session deletion. ' +
    'The current schema does not store source session or message provenance.'
);

function appendTextRow(container, label, value, { pre = false } = {}) {
    const row = document.createElement('div');
    row.className = 'knowledge-detail-row';
    const term = document.createElement('div');
    term.className = 'knowledge-detail-label';
    term.textContent = label;
    const content = document.createElement(pre ? 'pre' : 'div');
    content.className = pre ? 'knowledge-detail-value pre' : 'knowledge-detail-value';
    content.textContent = value === null || value === undefined || value === '' ? '—' : String(value);
    row.append(term, content);
    container.appendChild(row);
}

export class KnowledgeInspector {
    constructor(networkClient) {
        this.client = networkClient;
        this.activeSessionId = null;
        this.selectedSessionId = null;
        this.loadedSessions = false;
        this.view = 'effective';
        this.filters = { limit: 50 };
        this.offset = 0;
        this.currentTotal = 0;

        this.root = document.getElementById('knowledge-view');
        this.sessionSelect = document.getElementById('knowledge-session-select');
        this.sessionControl = document.querySelector('.knowledge-session-control');
        this.refreshButton = document.getElementById('knowledge-refresh-btn');
        this.status = document.getElementById('knowledge-status');
        this.subtabs = Array.from(document.querySelectorAll('.knowledge-subtab'));
        this.panels = Array.from(document.querySelectorAll('[data-knowledge-panel]'));
        this.effectiveTable = document.getElementById('knowledge-effective-table');
        this.allTable = document.getElementById('knowledge-all-table');
        this.contextText = document.getElementById('knowledge-context-text');
        this.memoryTable = document.getElementById('knowledge-memory-table');
        this.memoryCount = document.getElementById('knowledge-memory-count');
        this.memoryExplanation = document.getElementById('knowledge-memories-explanation');
        this.filterForm = document.getElementById('knowledge-filters');
        this.filterReset = document.getElementById('knowledge-filter-reset');
        this.previousButton = document.getElementById('knowledge-prev-btn');
        this.nextButton = document.getElementById('knowledge-next-btn');
        this.pageInfo = document.getElementById('knowledge-page-info');
        this.detail = document.getElementById('knowledge-detail');
        this.detailContent = document.getElementById('knowledge-detail-content');
        this.detailClose = document.getElementById('knowledge-detail-close');
        if (this.memoryExplanation) {
            this.memoryExplanation.textContent = SAVED_MEMORIES_EXPLANATION;
        }
        this._bind();
    }

    _bind() {
        document.querySelector('[data-tab-target="knowledge-view"]')?.addEventListener('click', () => {
            void this.activate();
        });
        this.refreshButton?.addEventListener('click', () => void this.refresh(true));
        this.sessionSelect?.addEventListener('change', () => {
            void this.selectInspectionSession(this.sessionSelect.value);
        });
        for (const tab of this.subtabs) {
            tab.addEventListener('click', () => {
                this.setView(tab.dataset.knowledgeView);
                void this.refresh(false);
            });
        }
        this.filterForm?.addEventListener('submit', (event) => {
            event.preventDefault();
            this.filters = readExactFilters(this.filterForm);
            this.offset = 0;
            void this.loadAll();
        });
        this.filterReset?.addEventListener('click', () => {
            this.filterForm.reset();
            this.filters = { limit: 50 };
            this.offset = 0;
            void this.loadAll();
        });
        this.previousButton?.addEventListener('click', () => {
            this.offset = Math.max(0, this.offset - Number(this.filters.limit || 50));
            void this.loadAll();
        });
        this.nextButton?.addEventListener('click', () => {
            this.offset += Number(this.filters.limit || 50);
            void this.loadAll();
        });
        this.detailClose?.addEventListener('click', () => this.closeDetail());
    }

    async activate() {
        if (!this.loadedSessions) await this.loadSessions();
        await this.refresh(false);
    }

    async setActiveSession(sessionId) {
        const previousActiveSessionId = this.activeSessionId;
        this.activeSessionId = sessionId || null;
        if (!this.selectedSessionId || this.selectedSessionId === previousActiveSessionId) {
            this.selectedSessionId = this.activeSessionId;
        }
        if (this.loadedSessions) await this.loadSessions();
    }

    async selectInspectionSession(sessionId) {
        this.selectedSessionId = sessionId || null;
        this.closeDetail();
        return this.refresh(false);
    }

    async loadSessions() {
        try {
            const payload = await this.client.listSessions();
            const ids = new Set((payload.sessions || []).map((item) => item.session_id));
            if (this.activeSessionId) ids.add(this.activeSessionId);
            const selected = this.selectedSessionId || this.activeSessionId || Array.from(ids)[0] || '';
            this.sessionSelect.replaceChildren();
            for (const sessionId of ids) {
                const option = document.createElement('option');
                option.value = sessionId;
                option.textContent = sessionId === this.activeSessionId
                    ? `${sessionId} (active)`
                    : sessionId;
                this.sessionSelect.appendChild(option);
            }
            this.selectedSessionId = ids.has(selected) ? selected : (this.activeSessionId || Array.from(ids)[0] || null);
            if (this.selectedSessionId) this.sessionSelect.value = this.selectedSessionId;
            this.loadedSessions = true;
            this.setStatus('');
        } catch (error) {
            this.setStatus('Failed to load inspection sessions.', 'error');
        }
    }

    setView(view) {
        if (!['effective', 'all', 'context', 'memories'].includes(view)) return;
        this.view = view;
        for (const tab of this.subtabs) {
            const active = tab.dataset.knowledgeView === view;
            tab.classList.toggle('active', active);
            tab.setAttribute('aria-selected', String(active));
        }
        for (const panel of this.panels) {
            const active = panel.dataset.knowledgePanel === view;
            panel.classList.toggle('active', active);
            panel.setAttribute('aria-hidden', String(!active));
        }
        this.sessionControl?.classList.toggle('hidden', view === 'memories');
        this.closeDetail();
    }

    async refresh(reloadSessions) {
        if (this.view === 'memories') return this.loadMemories();
        if (reloadSessions) await this.loadSessions();
        if (this.view === 'all') return this.loadAll();
        if (!this.selectedSessionId) {
            this.setStatus('No session is available for inspection.');
            return;
        }
        return this.view === 'context' ? this.loadContext() : this.loadEffective();
    }

    async loadEffective() {
        this.setLoading(true);
        try {
            const payload = await this.client.getEffectiveBeliefs(this.selectedSessionId);
            this.renderBeliefTable(this.effectiveTable, payload.records || []);
            this.setStatus(payload.records?.length ? '' : 'No effective beliefs for this session.');
        } catch (error) {
            this.setStatus('Failed to load effective beliefs.', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    async loadAll() {
        this.setLoading(true);
        try {
            const payload = await this.client.listBeliefs({
                ...this.filters,
                offset: this.offset,
            });
            this.currentTotal = payload.total || 0;
            this.renderBeliefTable(this.allTable, payload.records || []);
            this.renderPagination(payload.total, payload.limit, payload.offset);
            this.setStatus(payload.records?.length ? '' : 'No belief records match these filters.');
        } catch (error) {
            this.setStatus('Failed to load belief records.', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    async loadContext() {
        this.setLoading(true);
        try {
            const payload = await this.client.getBeliefContext(this.selectedSessionId);
            this.contextText.textContent = payload.text || '';
            this.setStatus(payload.state === 'empty' ? 'No belief context would be injected for this session.' : '');
        } catch (error) {
            this.contextText.textContent = '';
            this.setStatus('Failed to load belief context preview.', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    async loadMemories() {
        this.setLoading(true);
        this.setStatus('Loading saved memories…');
        try {
            const payload = await this.client.getSavedMemories();
            const records = payload.records || [];
            this.memoryCount.textContent = String(payload.total ?? records.length);
            this.renderSavedMemories(records);
            this.setStatus(records.length ? '' : 'No saved memories.');
        } catch (error) {
            this.memoryCount.textContent = 'Unavailable';
            this.memoryTable.replaceChildren();
            this.setStatus('Failed to load saved memories.', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    renderSavedMemories(records) {
        this.memoryTable.replaceChildren();
        if (!records.length) return;
        const scroll = document.createElement('div');
        scroll.className = 'knowledge-table-scroll';
        const table = document.createElement('table');
        table.className = 'knowledge-table knowledge-memory-table';
        const head = document.createElement('thead');
        const headRow = document.createElement('tr');
        for (const label of ['ID', 'Category', 'Content', 'Importance', 'Created', 'Last accessed']) {
            const cell = document.createElement('th');
            cell.textContent = label;
            headRow.appendChild(cell);
        }
        head.appendChild(headRow);
        const body = document.createElement('tbody');
        for (const record of records) {
            const row = document.createElement('tr');
            for (const value of [
                record.id,
                record.category ?? '',
                record.content,
                record.importance ?? '',
                record.created_at ?? '',
                record.last_accessed_at ?? '',
            ]) {
                const cell = document.createElement('td');
                cell.textContent = String(value);
                row.appendChild(cell);
            }
            body.appendChild(row);
        }
        table.append(head, body);
        scroll.appendChild(table);
        this.memoryTable.appendChild(scroll);
    }

    renderBeliefTable(container, records) {
        container.replaceChildren();
        if (!records.length) return;
        const scroll = document.createElement('div');
        scroll.className = 'knowledge-table-scroll';
        const table = document.createElement('table');
        table.className = 'knowledge-table';
        const head = document.createElement('thead');
        const headRow = document.createElement('tr');
        for (const label of ['Status', 'Subject', 'Predicate', 'Value', 'Epistemic', 'Source', 'Scope', 'Expiry', 'Revision', 'Updated']) {
            const cell = document.createElement('th');
            cell.textContent = label;
            headRow.appendChild(cell);
        }
        head.appendChild(headRow);
        const body = document.createElement('tbody');
        for (const record of records) {
            const row = document.createElement('tr');
            row.tabIndex = 0;
            const values = [
                record.record_status,
                `${record.subject?.display_name_at_evidence_time || ''} (${record.subject?.id || ''})`,
                record.predicate,
                record.value_json,
                record.epistemic_status,
                `${record.source?.display_name_at_evidence_time || ''} (${record.source?.sender_id || ''})`,
                record.visibility === 'SESSION_CURRENT' ? (record.scope_session_id || 'session') : 'global',
                record.expires_at || 'until revised',
                record.revision,
                record.updated_at,
            ];
            for (const value of values) {
                const cell = document.createElement('td');
                cell.textContent = String(value ?? '');
                row.appendChild(cell);
            }
            const open = () => void this.openDetail(record.belief_id);
            row.addEventListener('click', open);
            row.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') open();
            });
            body.appendChild(row);
        }
        table.append(head, body);
        scroll.appendChild(table);
        container.appendChild(scroll);
    }

    renderPagination(total, limit, offset) {
        const window = pageWindow(total, limit, offset);
        this.pageInfo.textContent = `${window.start}–${window.end} of ${total}`;
        this.previousButton.disabled = !window.hasPrevious;
        this.nextButton.disabled = !window.hasNext;
    }

    async openDetail(beliefId) {
        try {
            const detail = await this.client.getBelief(beliefId);
            this.renderDetail(detail);
            this.detail.classList.remove('hidden');
        } catch (error) {
            if (error?.status === 404) {
                this.closeDetail();
                await this.refresh(false);
                this.setStatus('That belief record no longer exists.', 'error');
                return;
            }
            this.setStatus('Failed to load belief details.', 'error');
        }
    }

    renderDetail(detail) {
        this.detailContent.replaceChildren();
        const fields = [
            ['Belief ID', detail.belief_id],
            ['Owner', detail.owner_agent_id],
            ['Record status', detail.record_status],
            ['Stored status', detail.stored_status],
            ['Subject display name at evidence time', detail.subject?.display_name_at_evidence_time],
            ['Subject ID', detail.subject?.id],
            ['Subject kind', detail.subject?.kind],
            ['Predicate', detail.predicate],
            ['Parsed JSON value', JSON.stringify(detail.value, null, 2), { pre: true }],
            ['Raw JSON value', detail.value_json, { pre: true }],
            ['Value parse status', detail.value_parse_error],
            ['Epistemic status', detail.epistemic_status],
            ['Source display name at evidence time', detail.source?.display_name_at_evidence_time],
            ['Source sender ID', detail.source?.sender_id],
            ['Source sender type', detail.source?.sender_type],
            ['Source input source', detail.source?.input_source],
            ['Visibility', detail.visibility],
            ['Scope session (controls lifetime)', detail.scope_session_id],
            ['Source session (provenance only)', detail.source_session_id],
            ['Source message ID', detail.source_message_id],
            ['Evidence excerpt', detail.evidence_excerpt],
            ['Confidence', detail.confidence],
            ['Expired', detail.is_expired],
            ['Expiry', detail.expires_at],
            ['Revision', detail.revision],
            ['Created', detail.created_at],
            ['Updated', detail.updated_at],
            ['Observed', detail.observed_at],
        ];
        for (const [label, value, options] of fields) {
            appendTextRow(this.detailContent, label, value, options);
        }
    }

    closeDetail() {
        this.detail?.classList.add('hidden');
        this.detailContent?.replaceChildren();
    }

    setStatus(message, tone = 'info') {
        this.status.textContent = message;
        this.status.classList.toggle('hidden', !message);
        this.status.classList.toggle('error', tone === 'error');
    }

    setLoading(loading) {
        this.refreshButton.disabled = loading;
        this.refreshButton.textContent = loading ? 'Loading…' : 'Refresh';
    }
}
