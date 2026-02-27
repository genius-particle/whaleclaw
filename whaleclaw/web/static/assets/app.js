/* WhaleClaw WebChat SPA — Vue 3 Composition API */

const { createApp, ref, reactive, computed, watch, nextTick, onMounted } = Vue;

/* ── markdown-it + highlight.js setup ── */
const md = window.markdownit({
  html: true,
  breaks: true,
  linkify: true,
  highlight(str, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        const html = hljs.highlight(str, { language: lang }).value;
        return `<pre class="hljs"><code>${html}</code><button class="copy-btn" onclick="copyCode(this)">复制</button></pre>`;
      } catch (_) { /* ignore */ }
    }
    return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code><button class="copy-btn" onclick="copyCode(this)">复制</button></pre>`;
  },
});

/* Rewrite local file paths in markdown image tokens to /api/local-file?path= */
const _defaultImageRender = md.renderer.rules.image ||
  function (tokens, idx, options, env, self) { return self.renderToken(tokens, idx, options); };

md.renderer.rules.image = function (tokens, idx, options, env, self) {
  const token = tokens[idx];
  const srcIdx = token.attrIndex('src');
  if (srcIdx >= 0) {
    let src = token.attrs[srcIdx][1];
    if (src && /^(\/|~\/|\.\/|\.\.\/)/.test(src) && !src.startsWith('/api/')) {
      token.attrs[srcIdx][1] = `/api/local-file?path=${encodeURIComponent(src)}`;
    } else if (src && src.includes('/api/local-file?path=')) {
      /* LLM already included the proxy prefix — ensure path portion is encoded */
      const parts = src.split('path=');
      if (parts.length === 2 && !parts[1].includes('%')) {
        token.attrs[srcIdx][1] = `${parts[0]}path=${encodeURIComponent(decodeURIComponent(parts[1]))}`;
      }
    }
  }
  return _defaultImageRender(tokens, idx, options, env, self);
};

window.copyCode = function (btn) {
  const code = btn.previousElementSibling.textContent;
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = '已复制';
    setTimeout(() => (btn.textContent = '复制'), 1500);
  });
};

/* ── App ── */
createApp({
  setup() {
    const theme = ref(
      localStorage.getItem('wc-theme') ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
    );
    watch(theme, (v) => {
      document.documentElement.setAttribute('data-theme', v);
      localStorage.setItem('wc-theme', v);
    }, { immediate: true });

    const token = ref(localStorage.getItem('wc-token') || '');
    const needLogin = ref(false);
    const authMode = ref('none');
    const loginPassword = ref('');
    const loginToken = ref('');
    const loginError = ref('');
    const compressionReady = ref(true);
    const compressionRunning = ref(false);

    const sessions = ref([]);
    const activeSessionId = ref('');
    const messages = ref([]);
    const inputText = ref('');
    const isStreaming = ref(false);
    const showSettings = ref(false);
    const showSidebar = ref(false);
    const pendingImages = ref([]);
    const memoryStyle = ref('');
    const memoryStyleEnabled = ref(true);
    const memoryStyleLoading = ref(false);
    const memoryStyleSaving = ref(false);

    const currentModel = ref('');
    const thinkingLevel = ref('off');
    const availableModels = ref([]);
    const defaultModel = ref('');
    const sessionTokens = ref({ input_tokens: 0, output_tokens: 0 });
    const totalTokens = ref({ input_tokens: 0, output_tokens: 0 });

    const activeTab = ref('chat');
    const skills = ref([]);
    const tools = ref([]);
    const skillInstallSource = ref('');
    const skillInstalling = ref(false);
    const skillDetail = ref(null);
    const toolDetail = ref(null);

    const activeSession = computed(() =>
      sessions.value.find((s) => s.id === activeSessionId.value)
    );

    function _bumpMessageCount() {
      const s = sessions.value.find((s) => s.id === activeSessionId.value);
      if (s) s.message_count = messages.value.length;
    }

    const _PROVIDER_LABELS = {
      anthropic: 'Anthropic', openai: 'OpenAI', deepseek: 'DeepSeek',
      qwen: '通义千问', zhipu: '智谱 GLM', minimax: 'MiniMax',
      moonshot: '月之暗面', google: 'Google', nvidia: 'NVIDIA NIM',
    };

    const groupedModels = computed(() => {
      const groups = {};
      for (const m of availableModels.value) {
        const p = m.provider || 'other';
        if (!groups[p]) groups[p] = [];
        groups[p].push(m);
      }
      return Object.entries(groups).map(([provider, models]) => ({
        provider,
        label: _PROVIDER_LABELS[provider] || provider,
        models,
      }));
    });

    let ws = null;
    let streamingMessage = null;
    let _compressionPollTimer = null;

    /* ── API helpers ── */
    const apiBase = window.location.origin;

    async function apiFetch(path, opts = {}) {
      const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
      if (token.value) headers['Authorization'] = `Bearer ${token.value}`;
      const res = await fetch(`${apiBase}${path}`, { ...opts, headers });
      if (res.status === 401) {
        if (!needLogin.value) needLogin.value = true;
        throw new Error('auth');
      }
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || data.detail || `请求失败 (${res.status})`);
      }
      return data;
    }

    /* ── Auth ── */
    async function checkAuth() {
      try {
        await apiFetch('/api/auth/verify');
        needLogin.value = false;
      } catch {
        /* needLogin already set by apiFetch */
      }
    }

    async function doLogin() {
      loginError.value = '';

      if (authMode.value === 'token') {
        const t = loginToken.value.trim();
        if (!t) { loginError.value = '请输入 Token'; return; }
        token.value = t;
        localStorage.setItem('wc-token', t);
        try {
          await apiFetch('/api/auth/verify');
          needLogin.value = false;
          await init();
        } catch {
          token.value = '';
          localStorage.removeItem('wc-token');
          loginError.value = 'Token 无效';
        }
        return;
      }

      try {
        const res = await fetch(`${apiBase}/api/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password: loginPassword.value }),
        });
        const data = await res.json();
        if (data.token) {
          token.value = data.token;
          localStorage.setItem('wc-token', data.token);
          needLogin.value = false;
          await init();
        } else {
          loginError.value = data.error || '登录失败';
        }
      } catch (e) {
        loginError.value = '网络错误';
      }
    }

    function doLogout() {
      token.value = '';
      localStorage.removeItem('wc-token');
      loginPassword.value = '';
      loginToken.value = '';
      loginError.value = '';
      needLogin.value = true;
      if (ws) {
        ws._intentionalClose = true;
        ws.close();
        ws = null;
      }
    }

    /* ── Models ── */
    async function loadModels() {
      try {
        const data = await apiFetch('/api/models');
        availableModels.value = data.models || [];
        defaultModel.value = data.default || '';
        if (!currentModel.value && defaultModel.value) {
          currentModel.value = defaultModel.value;
        }
        if (data.thinking_level) {
          thinkingLevel.value = data.thinking_level;
        }
      } catch { /* ignore */ }
    }

    let _tokenUsageTimer = null;
    function loadTokenUsage(sessionId) {
      if (needLogin.value) return;
      if (!token.value && authMode.value !== 'none') return;
      clearTimeout(_tokenUsageTimer);
      _tokenUsageTimer = setTimeout(async () => {
        if (needLogin.value) return;
        try {
          const [sData, tData] = await Promise.all([
            apiFetch(`/api/sessions/${sessionId}/token-usage`),
            apiFetch('/api/token-usage'),
          ]);
          sessionTokens.value = sData;
          totalTokens.value = tData.total || { input_tokens: 0, output_tokens: 0 };
        } catch { /* ignore */ }
      }, 500);
    }

    /* ── Skills & Tools ── */
    async function loadSkills() {
      try {
        skills.value = await apiFetch('/api/skills');
      } catch { /* ignore */ }
    }

    async function loadTools() {
      try {
        tools.value = await apiFetch('/api/tools');
      } catch { /* ignore */ }
    }

    /* ── Global Memory Style ── */
    async function loadMemoryStyle() {
      memoryStyleLoading.value = true;
      try {
        const data = await apiFetch('/api/memory/style');
        memoryStyleEnabled.value = data.enabled !== false;
        memoryStyle.value = data.style_directive || '';
      } catch { /* ignore */ }
      finally {
        memoryStyleLoading.value = false;
      }
    }

    async function saveMemoryStyle() {
      if (!memoryStyle.value.trim()) {
        alert('风格指令不能为空');
        return;
      }
      memoryStyleSaving.value = true;
      try {
        await apiFetch('/api/memory/style', {
          method: 'POST',
          body: JSON.stringify({ style_directive: memoryStyle.value.trim() }),
        });
      } catch (e) {
        alert('保存失败: ' + (e.message || e));
      } finally {
        memoryStyleSaving.value = false;
      }
    }

    async function clearMemoryStyle() {
      if (!confirm('确定清除全局回复风格吗？')) return;
      memoryStyleSaving.value = true;
      try {
        await apiFetch('/api/memory/style', { method: 'DELETE' });
        memoryStyle.value = '';
      } catch (e) {
        alert('清除失败: ' + (e.message || e));
      } finally {
        memoryStyleSaving.value = false;
      }
    }

    const _SKILL_CATEGORIES = {
      bundled: '内置技能',
      user: '已安装技能',
    };

    const groupedSkills = computed(() => {
      const groups = {};
      for (const s of skills.value) {
        const src = s.source || 'bundled';
        if (!groups[src]) groups[src] = { label: _SKILL_CATEGORIES[src] || src, items: [] };
        groups[src].items.push(s);
      }
      return Object.values(groups);
    });

    const _TOOL_CATEGORY_LABELS = {
      system: '系统',
      file: '文件操作',
      browser: '浏览器',
      session: '会话管理',
      automation: '自动化',
      other: '其他',
    };
    const _TOOL_CATEGORY_ORDER = ['system', 'file', 'browser', 'session', 'automation', 'other'];

    const groupedTools = computed(() => {
      const groups = {};
      for (const t of tools.value) {
        const cat = t.category || 'other';
        if (!groups[cat]) groups[cat] = { label: _TOOL_CATEGORY_LABELS[cat] || cat, items: [] };
        groups[cat].items.push(t);
      }
      return _TOOL_CATEGORY_ORDER
        .filter((k) => groups[k])
        .map((k) => groups[k]);
    });

    async function installSkill() {
      const src = skillInstallSource.value.trim();
      if (!src) return;
      skillInstalling.value = true;
      try {
        await apiFetch('/api/skills/install', {
          method: 'POST',
          body: JSON.stringify({ source: src }),
        });
        skillInstallSource.value = '';
        await loadSkills();
      } catch (e) {
        alert('安装失败: ' + (e.message || e));
      } finally {
        skillInstalling.value = false;
      }
    }

    async function uninstallSkill(skillId) {
      if (!confirm(`确定卸载技能「${skillId}」?`)) return;
      try {
        await apiFetch(`/api/skills/${skillId}`, { method: 'DELETE' });
        await loadSkills();
      } catch (e) {
        alert('卸载失败: ' + (e.message || e));
      }
    }

    async function showSkillDetail(skillId) {
      try {
        const data = await apiFetch(`/api/skills/${skillId}`);
        skillDetail.value = data;
      } catch (e) {
        skillDetail.value = null;
      }
    }

    function formatTokens(n) {
      if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
      if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
      return String(n);
    }

    /* ── Sessions ── */
    async function loadSessions() {
      try {
        sessions.value = await apiFetch('/api/sessions');
      } catch { /* ignore auth redirect */ }
    }

    async function createSession() {
      const data = await apiFetch('/api/sessions', { method: 'POST' });
      await loadSessions();
      await switchSession(data.id);
    }

    async function deleteSession(id) {
      await apiFetch(`/api/sessions/${id}`, { method: 'DELETE' });
      if (activeSessionId.value === id) {
        activeSessionId.value = '';
        messages.value = [];
      }
      await loadSessions();
    }

    async function switchSession(id) {
      activeSessionId.value = id;
      try {
        const data = await apiFetch(`/api/sessions/${id}`);
        messages.value = (data.messages || []).map((m, i) => ({
          id: `hist-${i}`,
          role: m.role,
          content: m.content,
          rendered: renderMarkdown(m.content),
          toolCalls: [],
        }));
        currentModel.value = data.model || '';
        thinkingLevel.value = data.thinking_level || 'off';
      } catch { /* ignore */ }
      loadTokenUsage(id);
      connectWS(id);
      await nextTick();
      scrollToBottom();
    }

    /* ── WebSocket ── */
    let _wsGeneration = 0;

    function connectWS(sessionId) {
      if (ws) {
        ws._intentionalClose = true;
        ws.close();
        ws = null;
      }
      const gen = ++_wsGeneration;
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      let url = `${protocol}//${location.host}/ws`;
      if (token.value) url += `?token=${encodeURIComponent(token.value)}`;
      const socket = new WebSocket(url);
      ws = socket;

      socket.onopen = () => {
        socket._ping = setInterval(() => {
          if (socket.readyState === 1) {
            socket.send(JSON.stringify({ type: 'ping' }));
          }
        }, 25000);
      };

      socket.onmessage = (evt) => {
        if (gen !== _wsGeneration) return;
        let msg;
        try { msg = JSON.parse(evt.data); } catch { return; }

        if (msg.type === 'stream') {
          if (!streamingMessage) {
            streamingMessage = {
              id: `msg-${Date.now()}`,
              role: 'assistant',
              content: '',
              rendered: '',
              toolCalls: [],
            };
            messages.value.push(streamingMessage);
          }
          streamingMessage.content += msg.payload.content || '';
          streamingMessage.rendered = renderMarkdown(streamingMessage.content);
          scrollToBottom();
        } else if (msg.type === 'message') {
          const msgSid = msg.session_id || '';
          const isOwnSession = msgSid && msgSid === activeSessionId.value;
          isStreaming.value = false;
          if (streamingMessage) {
            streamingMessage.content = msg.payload.content || streamingMessage.content;
            streamingMessage.rendered = renderMarkdown(streamingMessage.content);
            streamingMessage = null;
          } else if (isOwnSession) {
            messages.value.push({
              id: `msg-${Date.now()}`,
              role: 'assistant',
              content: msg.payload.content,
              rendered: renderMarkdown(msg.payload.content),
              toolCalls: [],
            });
          }
          _bumpMessageCount();
          if (activeSessionId.value) loadTokenUsage(activeSessionId.value);
          scrollToBottom();
        } else if (msg.type === 'tool_call') {
          if (!streamingMessage) {
            streamingMessage = {
              id: `msg-${Date.now()}`,
              role: 'assistant',
              content: '',
              rendered: '',
              toolCalls: [],
            };
            messages.value.push(streamingMessage);
            isStreaming.value = true;
          }
          const tc = {
            name: msg.payload.name,
            args: JSON.stringify(msg.payload.arguments, null, 2),
            result: null,
            loading: true,
            collapsed: false,
          };
          streamingMessage.toolCalls.push(tc);
          scrollToBottom();
        } else if (msg.type === 'tool_result') {
          if (streamingMessage) {
            const tc = streamingMessage.toolCalls.find(
              (t) => t.name === msg.payload.name && t.loading
            );
            if (tc) {
              tc.result = msg.payload.output;
              tc.loading = false;
              tc.collapsed = true;
            }
            scrollToBottom();
          }
        } else if (msg.type === 'error') {
          isStreaming.value = false;
          streamingMessage = null;
          messages.value.push({
            id: `err-${Date.now()}`,
            role: 'assistant',
            content: `**Error:** ${msg.payload.error}`,
            rendered: renderMarkdown(`**Error:** ${msg.payload.error}`),
            toolCalls: [],
          });
          scrollToBottom();
        } else if (msg.type === 'status') {
          const text = msg.payload?.text || '';
          const model = msg.payload?.model || '';
          const sid = msg.session_id || activeSessionId.value;
          if (model) {
            if (sid && sid === activeSessionId.value) {
              currentModel.value = model;
            }
            const s = sessions.value.find((x) => x.id === sid);
            if (s) s.model = model;
          }
          if (text && sid && sid === activeSessionId.value) {
            messages.value.push({
              id: `status-${Date.now()}`,
              role: 'assistant',
              content: text,
              rendered: renderMarkdown(text),
              toolCalls: [],
            });
            _bumpMessageCount();
            scrollToBottom();
          }
        }
      };

      socket.onclose = () => {
        clearInterval(socket._ping);
        if (socket._intentionalClose) return;
        if (gen !== _wsGeneration) return;
        setTimeout(() => {
          if (gen === _wsGeneration && activeSessionId.value === sessionId) {
            connectWS(sessionId);
          }
        }, 2000);
      };
    }

    /* ── Send message ── */
    function sendMessage() {
      const text = inputText.value.trim();
      const imgs = pendingImages.value;
      if (!compressionReady.value) return;
      if ((!text && !imgs.length) || !ws || ws.readyState !== 1) return;

      let displayHtml = renderMarkdown(text);
      if (imgs.length) {
        const imgHtml = imgs.map((img) => `<img src="${img.dataUrl}" style="max-width:200px;max-height:160px;border-radius:6px;margin:4px 2px">`).join('');
        displayHtml = imgHtml + (displayHtml ? '<br>' + displayHtml : '');
      }

      messages.value.push({
        id: `user-${Date.now()}`,
        role: 'user',
        content: text,
        rendered: displayHtml,
        toolCalls: [],
      });

      const payload = { content: text || '(用户发送了图片)' };
      if (imgs.length) {
        payload.images = imgs.map((img) => ({
          data: img.dataUrl.split(',')[1],
          mime: img.mime,
          name: img.name,
        }));
      }

      ws.send(JSON.stringify({
        type: 'message',
        session_id: activeSessionId.value,
        payload,
      }));

      inputText.value = '';
      pendingImages.value = [];
      isStreaming.value = true;
      streamingMessage = null;
      _bumpMessageCount();
      nextTick(scrollToBottom);
    }

    function handleKeydown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }

    /* ── Image handling ── */
    function addImageFiles(files) {
      for (const file of files) {
        if (!file.type.startsWith('image/')) continue;
        if (pendingImages.value.length >= 4) break;
        const reader = new FileReader();
        reader.onload = (e) => {
          pendingImages.value.push({
            dataUrl: e.target.result,
            mime: file.type,
            name: file.name || 'image.png',
          });
        };
        reader.readAsDataURL(file);
      }
    }

    function removeImage(idx) {
      pendingImages.value.splice(idx, 1);
    }

    function onPaste(e) {
      const items = e.clipboardData?.items;
      if (!items) return;
      const imageFiles = [];
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const f = item.getAsFile();
          if (f) imageFiles.push(f);
        }
      }
      if (imageFiles.length) {
        e.preventDefault();
        addImageFiles(imageFiles);
      }
    }

    function onDrop(e) {
      e.preventDefault();
      if (e.dataTransfer?.files) addImageFiles(e.dataTransfer.files);
    }

    function onDragOver(e) { e.preventDefault(); }

    function triggerFileInput() {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.multiple = true;
      input.onchange = (e) => { if (e.target.files) addImageFiles(e.target.files); };
      input.click();
    }

    /* ── Settings ── */
    async function switchModel(model) {
      if (model === currentModel.value) return;
      currentModel.value = model;
      const modelInfo = availableModels.value.find((m) => m.id === model);
      if (modelInfo && modelInfo.thinking) {
        thinkingLevel.value = modelInfo.thinking;
      }
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({
          type: 'message',
          session_id: activeSessionId.value,
          payload: { content: `/model ${model}` },
        }));
        if (modelInfo && modelInfo.thinking && modelInfo.thinking !== 'off') {
          ws.send(JSON.stringify({
            type: 'message',
            session_id: activeSessionId.value,
            payload: { content: `/thinking ${modelInfo.thinking}` },
          }));
        }
      }
    }

    /* ── Utils ── */
    const _FILE_ICONS = {
      pptx: '📊', ppt: '📊',
      xlsx: '📗', xls: '📗', csv: '📗',
      docx: '📝', doc: '📝',
      pdf: '📕',
      zip: '📦', rar: '📦', '7z': '📦', tar: '📦', gz: '📦',
      mp3: '🎵', wav: '🎵', flac: '🎵',
      mp4: '🎬', mov: '🎬', avi: '🎬',
      py: '🐍', js: '📜', ts: '📜', json: '📜',
      txt: '📄', md: '📄', log: '📄',
    };
    const _IMAGE_EXTS = new Set(['jpg','jpeg','png','gif','webp','bmp','svg','ico']);

    function _appendToken(url) {
      if (!token.value) return url;
      const sep = url.includes('?') ? '&' : '?';
      return `${url}${sep}token=${encodeURIComponent(token.value)}`;
    }

    function _fileCard(filePath) {
      const name = filePath.split('/').pop();
      const ext = (name.split('.').pop() || '').toLowerCase();
      const icon = _FILE_ICONS[ext] || '📎';
      const encPath = encodeURIComponent(filePath);
      const downloadUrl = _appendToken(`/api/local-file?path=${encPath}&download=true`);
      const openUrl = _appendToken(`/api/local-file?path=${encPath}`);
      return `<div class="file-card" onclick="window.open('${openUrl}','_blank')">` +
        `<div class="file-card-icon">${icon}</div>` +
        `<div class="file-card-info">` +
          `<div class="file-card-name">${name}</div>` +
          `<div class="file-card-meta" data-path="${encPath}">加载中...</div>` +
        `</div>` +
        `<a class="file-card-dl" href="${downloadUrl}" onclick="event.stopPropagation()" title="下载">⬇</a>` +
      `</div>`;
    }

    let _metaTimer = null;
    function _loadFileCardMeta() {
      clearTimeout(_metaTimer);
      _metaTimer = setTimeout(() => {
        document.querySelectorAll('.file-card-meta[data-path]').forEach(async (el) => {
          const p = el.dataset.path;
          if (!p || el.dataset.loaded) return;
          el.dataset.loaded = '1';
          try {
            const resp = await fetch(_appendToken(`/api/file-info?path=${p}`));
            if (resp.ok) {
              const info = await resp.json();
              el.textContent = `${info.ext.toUpperCase()} 文件 · ${info.size_human}`;
            } else {
              el.textContent = '文件不可用';
            }
          } catch { el.textContent = ''; }
        });
      }, 300);
    }

    function renderMarkdown(text) {
      if (!text) return '';
      let html = md.render(text);

      /* Rewrite <img src="/local/path"> to use /api/local-file proxy */
      html = html.replace(
        /(<img\s[^>]*src=["'])(\/(tmp|home|Users|var|opt|etc)[^"']+)(["'][^>]*>)/gi,
        (m, pre, src, _d, post) => {
          if (m.includes('/api/local-file')) return m;
          return `${pre}/api/local-file?path=${encodeURIComponent(src)}${post}`;
        }
      );

      /* Convert file paths to file cards (non-image files only) */
      html = html.replace(
        /(\/(?:tmp|home|Users|var|opt|etc)\/[^\s<"']*\.(\w{2,5}))(?=[\s<"']|$)/gi,
        (m, filePath, ext) => {
          if (_IMAGE_EXTS.has(ext.toLowerCase())) return m;
          if (m.includes('file-card')) return m;
          return _fileCard(filePath);
        }
      );

      /* Append auth token to /api/ URLs so <img> and file cards pass auth */
      if (token.value) {
        html = html.replace(
          /((?:src|href)=["'])(\/api\/[^"']+)(["'])/gi,
          (m, pre, url, post) => `${pre}${_appendToken(url)}${post}`
        );
      }

      nextTick(_loadFileCardMeta);
      return html;
    }

    const messagesEl = ref(null);
    function scrollToBottom() {
      nextTick(() => {
        if (messagesEl.value) {
          messagesEl.value.scrollTop = messagesEl.value.scrollHeight;
        }
      });
    }

    function toggleTheme() {
      theme.value = theme.value === 'dark' ? 'light' : 'dark';
    }

    function formatTime(iso) {
      if (!iso) return '';
      const d = new Date(iso);
      return d.toLocaleDateString();
    }

    /* ── Init ── */
    async function init() {
      await loadModels();
      await loadSessions();
      if (sessions.value.length > 0) {
        await switchSession(sessions.value[0].id);
      } else {
        await createSession();
      }
      loadSkills();
      loadTools();
      loadMemoryStyle();
    }

    onMounted(async () => {
      const updateCompressionState = (status) => {
        compressionReady.value = status.compression_ready !== false;
        compressionRunning.value = status.compression_running === true;
      };
      const pollCompressionState = () => {
        clearTimeout(_compressionPollTimer);
        if (compressionReady.value) return;
        _compressionPollTimer = setTimeout(async () => {
          try {
            const s = await fetch(`${apiBase}/api/status`).then((r) => r.json());
            updateCompressionState(s);
          } catch { /* ignore */ }
          pollCompressionState();
        }, 3000);
      };
      try {
        const status = await fetch(`${apiBase}/api/status`).then((r) => r.json());
        updateCompressionState(status);
        if (status.status === 'ok') {
          authMode.value = status.auth_mode || 'none';
          if (authMode.value === 'none') {
            needLogin.value = false;
            await init();
          } else {
            await checkAuth();
            if (!needLogin.value) await init();
          }
        }
        pollCompressionState();
      } catch {
        /* Gateway unreachable — assume no auth, try to init */
        try { await init(); } catch { /* ignore */ }
      }
    });

    return {
      theme, token, needLogin, authMode, loginPassword, loginToken, loginError, doLogin, doLogout,
      compressionReady, compressionRunning,
      sessions, activeSessionId, activeSession, messages,
      inputText, isStreaming, showSettings, showSidebar, pendingImages,
      currentModel, thinkingLevel, availableModels, defaultModel, groupedModels, messagesEl,
      sessionTokens, totalTokens, formatTokens,
      activeTab, skills, tools, groupedSkills, groupedTools,
      skillInstallSource, skillInstalling, skillDetail, installSkill, uninstallSkill, showSkillDetail,
      toolDetail,
      memoryStyle, memoryStyleEnabled, memoryStyleLoading, memoryStyleSaving,
      loadMemoryStyle, saveMemoryStyle, clearMemoryStyle,
      createSession, deleteSession, switchSession,
      sendMessage, handleKeydown, switchModel, loadModels,
      toggleTheme, formatTime, renderMarkdown,
      addImageFiles, removeImage, onPaste, onDrop, onDragOver, triggerFileInput,
    };
  },

  template: `
    <!-- Login Overlay -->
    <div v-if="needLogin" class="login-overlay">
      <div class="login-card">
        <h2>WhaleClaw</h2>
        <p v-if="loginError" class="login-error">{{ loginError }}</p>
        <template v-if="authMode === 'token'">
          <input
            v-model="loginToken"
            type="password"
            placeholder="输入 Token"
            @keydown.enter="doLogin"
          />
        </template>
        <template v-else>
          <input
            v-model="loginPassword"
            type="password"
            placeholder="输入密码"
            @keydown.enter="doLogin"
          />
        </template>
        <button class="btn-send" @click="doLogin">登录</button>
      </div>
    </div>

    <!-- Main App -->
    <template v-else>
      <!-- Sidebar -->
      <aside class="sidebar" :class="{ open: showSidebar }">
        <div class="sidebar-header">
          <h1>WhaleClaw</h1>
          <div class="sidebar-actions">
            <button class="btn-icon" @click="createSession" title="新建会话">+</button>
            <button class="btn-icon" @click="toggleTheme" title="切换主题">
              {{ theme === 'dark' ? '☀' : '🌙' }}
            </button>
            <button class="btn-icon" @click="showSettings = !showSettings" title="设置">⚙</button>
          </div>
        </div>
        <div class="session-list" v-show="activeTab === 'chat'">
          <div
            v-for="s in sessions"
            :key="s.id"
            class="session-item"
            :class="{ active: s.id === activeSessionId }"
            @click="switchSession(s.id); showSidebar = false"
          >
            <div class="session-info">
              <div class="session-title">{{ s.model || '会话' }}</div>
              <div class="session-meta">{{ formatTime(s.created_at) }} · {{ s.message_count || 0 }} 条<template v-if="s.tokens"> · 🪙{{ formatTokens(s.tokens) }}</template></div>
            </div>
            <button class="session-delete" @click.stop="deleteSession(s.id)">✕</button>
          </div>
        </div>
        <div class="sidebar-tab-bar">
          <button class="sidebar-tab" :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'">💬 聊天</button>
          <button class="sidebar-tab" :class="{ active: activeTab === 'skills' }" @click="activeTab = 'skills'; loadSkills()">🧩 技能</button>
          <button class="sidebar-tab" :class="{ active: activeTab === 'tools' }" @click="activeTab = 'tools'; loadTools()">🔧 工具</button>
        </div>
      </aside>

      <!-- Main Area -->
      <div class="main">
        <!-- Chat Header (only in chat tab) -->
        <div class="chat-header" v-if="activeTab === 'chat'">
          <div style="display:flex;align-items:center;gap:8px">
            <button class="btn-icon mobile-menu" @click="showSidebar = !showSidebar">☰</button>
            <select
              class="model-selector"
              :value="currentModel"
              @change="switchModel($event.target.value)"
            >
              <template v-if="groupedModels.length">
                <optgroup v-for="g in groupedModels" :key="g.provider" :label="g.label">
                  <option
                    v-for="m in g.models"
                    :key="m.id"
                    :value="m.id"
                  >{{ m.name }}{{ m.thinking && m.thinking !== 'off' ? ' 💭' : '' }}{{ m.tools ? '' : ' ⚠无工具' }}</option>
                </optgroup>
              </template>
              <template v-else>
                <option :value="currentModel">{{ currentModel || '未配置模型' }}</option>
              </template>
            </select>
            <span v-if="thinkingLevel !== 'off'" class="thinking-badge">💭 {{ thinkingLevel }}</span>
          </div>
          <div class="chat-header-info" v-if="activeSession">
            <span>{{ activeSession.message_count || messages.length }} 条消息</span>
            <span class="token-badge"
              :title="'本会话: ↑' + sessionTokens.input_tokens.toLocaleString() + ' ↓' + sessionTokens.output_tokens.toLocaleString() + '\\n总计: ↑' + totalTokens.input_tokens.toLocaleString() + ' ↓' + totalTokens.output_tokens.toLocaleString()"
            >🪙 ↑{{ formatTokens(sessionTokens.input_tokens) }} ↓{{ formatTokens(sessionTokens.output_tokens) }} · 总{{ formatTokens(totalTokens.input_tokens + totalTokens.output_tokens) }}</span>
          </div>
        </div>

        <!-- Page Header (skills / tools) -->
        <div class="page-header" v-if="activeTab !== 'chat'">
          <button class="btn-icon mobile-menu" @click="showSidebar = !showSidebar">☰</button>
          <h2 v-if="activeTab === 'skills'">🧩 技能管理</h2>
          <h2 v-if="activeTab === 'tools'">🔧 工具列表 <small>({{ tools.length }})</small></h2>
        </div>

        <!-- Chat Tab -->
        <template v-if="activeTab === 'chat'">
          <div class="messages" ref="messagesEl">
            <div
              v-for="msg in messages"
              :key="msg.id"
              class="message-row"
              :class="msg.role"
            >
              <div class="bubble" :class="msg.role">
                <div v-if="msg.rendered" v-html="msg.rendered"></div>
                <div v-if="msg.toolCalls && msg.toolCalls.length" class="tool-calls">
                  <div v-for="(tc, ti) in msg.toolCalls" :key="ti" class="tool-card" :class="{ loading: tc.loading }">
                    <div class="tool-card-header" @click="tc.collapsed = !tc.collapsed">
                      <span class="tool-card-icon">{{ tc.loading ? '⏳' : '✅' }}</span>
                      <span class="tool-card-name">{{ tc.name }}</span>
                      <span class="tool-card-status">{{ tc.loading ? '执行中...' : '完成' }}</span>
                      <span class="tool-card-toggle">{{ tc.collapsed ? '▸' : '▾' }}</span>
                    </div>
                    <div v-if="!tc.collapsed" class="tool-card-body">
                      <pre class="tool-card-args">{{ tc.args }}</pre>
                      <pre v-if="tc.result" class="tool-card-result">{{ tc.result.length > 500 ? tc.result.slice(0, 500) + '...' : tc.result }}</pre>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div v-if="isStreaming && !messages.length" class="message-row assistant">
              <div class="bubble assistant">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
              </div>
            </div>
          </div>

          <div class="input-area" @drop="onDrop" @dragover="onDragOver">
            <div v-if="!compressionReady" style="font-size:12px;color:#f59e0b;padding:0 12px 8px;">
              {{ compressionRunning ? '会话压缩中，请稍后发送消息…' : '压缩尚未就绪，请稍后重试。' }}
            </div>
            <div v-if="pendingImages.length" class="image-preview-strip">
              <div v-for="(img, idx) in pendingImages" :key="idx" class="image-preview-item">
                <img :src="img.dataUrl" />
                <button class="image-remove-btn" @click="removeImage(idx)">✕</button>
              </div>
            </div>
            <div class="input-wrapper">
              <button class="btn-attach" @click="triggerFileInput" title="添加图片">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
              </button>
              <textarea
                v-model="inputText"
                placeholder="输入消息... (Enter 发送, Shift+Enter 换行, 可粘贴/拖拽图片)"
                @keydown="handleKeydown"
                @paste="onPaste"
                :disabled="!compressionReady"
                rows="1"
              ></textarea>
              <button class="btn-send" :disabled="!compressionReady || isStreaming || (!inputText.trim() && !pendingImages.length)" @click="sendMessage">
                发送
              </button>
            </div>
          </div>
        </template>

        <!-- Skills Tab -->
        <template v-if="activeTab === 'skills'">
          <div class="tab-content">
            <div class="panel-toolbar">
              <input
                v-model="skillInstallSource"
                class="panel-search"
                placeholder="输入 GitHub 地址或本地路径安装技能…"
                @keydown.enter="installSkill"
              />
              <button class="btn-pill" :disabled="skillInstalling || !skillInstallSource.trim()" @click="installSkill">
                {{ skillInstalling ? '安装中…' : '+ 安装' }}
              </button>
            </div>
            <div class="tab-content-body">
              <div v-if="!skills.length" class="tab-empty">暂无技能</div>
              <details v-for="group in groupedSkills" :key="group.label" class="panel-group" open>
                <summary class="panel-group-header">
                  <span>{{ group.label }}</span>
                  <span class="panel-group-count">{{ group.items.length }}</span>
                </summary>
                <div class="panel-grid skills-grid">
                  <div v-for="s in group.items" :key="s.id" class="panel-card skill-card" @click="showSkillDetail(s.id)">
                    <div class="panel-card-main">
                      <div class="panel-card-title">🧩 {{ s.name }}</div>
                      <div v-if="s.trigger_description" class="panel-card-desc">{{ s.trigger_description }}</div>
                      <div class="chip-row">
                        <span class="chip" :class="s.source === 'user' ? 'chip-accent' : 'chip-ok'">{{ s.source === 'user' ? '已安装' : '内置' }}</span>
                        <span class="chip" v-for="t in (s.triggers || []).slice(0, 3)" :key="t">{{ t }}</span>
                      </div>
                      <div v-if="s.tools && s.tools.length" class="panel-card-tools">
                        <code v-for="t in s.tools" :key="t" class="tool-code">{{ t }}</code>
                      </div>
                    </div>
                    <div class="panel-card-action">
                      <button v-if="s.source === 'user'" class="btn-danger-sm" @click.stop="uninstallSkill(s.id)" title="卸载">✕</button>
                    </div>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </template>

        <!-- Tools Tab -->
        <template v-if="activeTab === 'tools'">
          <div class="tab-content">
            <div class="tab-content-body">
              <div v-if="!tools.length" class="tab-empty">暂无工具</div>
              <details v-for="group in groupedTools" :key="group.label" class="panel-group" open>
                <summary class="panel-group-header">
                  <span>{{ group.label }}</span>
                  <span class="panel-group-count">{{ group.items.length }}</span>
                </summary>
                <div class="panel-grid tools-grid">
                  <div v-for="t in group.items" :key="t.name" class="panel-card tool-item tool-card" @click="toolDetail = t">
                    <div class="panel-card-main">
                      <div class="panel-card-title"><code class="tool-name-code">{{ t.name }}</code></div>
                      <div class="panel-card-desc">{{ t.description }}</div>
                    </div>
                    <div v-if="t.parameters && t.parameters.length" class="tool-params-strip">
                      <span v-for="p in t.parameters" :key="p.name" class="tool-param-chip" :class="{ required: p.required }">{{ p.name }}<em>{{ p.type }}</em></span>
                    </div>
                  </div>
                </div>
              </details>
            </div>
          </div>
        </template>
      </div>

      <!-- Settings Panel -->
      <div class="settings-panel" :class="{ open: showSettings }">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
          <h3>设置</h3>
          <button class="btn-icon" @click="showSettings = false">✕</button>
        </div>
        <div class="setting-group">
          <label>当前模型</label>
          <select :value="currentModel" @change="switchModel($event.target.value)">
            <template v-if="groupedModels.length">
              <optgroup v-for="g in groupedModels" :key="g.provider" :label="g.label">
                <option v-for="m in g.models" :key="m.id" :value="m.id">
                  {{ m.name }}{{ m.tools ? '' : ' ⚠无工具' }}
                </option>
              </optgroup>
            </template>
            <template v-else>
              <option :value="currentModel">{{ currentModel || '未配置' }}</option>
            </template>
          </select>
          <p style="font-size:12px;color:var(--text-secondary);margin-top:4px">
            在「修改配置.command」中添加更多 API Key 后刷新页面
          </p>
        </div>
        <div class="setting-group">
          <label>主题</label>
          <select :value="theme" @change="theme = $event.target.value">
            <option value="light">亮色</option>
            <option value="dark">暗色</option>
          </select>
        </div>
        <div class="setting-group">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <label style="margin:0">全局回复风格（自动生成，可手动覆盖）</label>
            <button class="btn-mini" @click="loadMemoryStyle" :disabled="memoryStyleLoading">刷新</button>
          </div>
          <p class="setting-hint" v-if="memoryStyleEnabled">系统会根据多轮对话自动提炼并填入；你也可以在这里手动修正。每轮默认应用，用户本轮明确要求优先。</p>
          <p class="setting-hint" v-else>全局风格功能已关闭（config: agent.memory.global_style_enabled=false）。</p>
          <textarea
            class="setting-textarea"
            v-model="memoryStyle"
            rows="4"
            placeholder="自动生成后会显示在这里；也可手动输入，例如：回答风格：简洁明了，先结论后细节，优先要点列表。"
          ></textarea>
          <div class="setting-actions">
            <button class="btn-pill" @click="saveMemoryStyle" :disabled="memoryStyleSaving || !memoryStyleEnabled || !memoryStyle.trim()">
              {{ memoryStyleSaving ? '保存中…' : '手动覆盖保存' }}
            </button>
            <button class="btn-outline" @click="clearMemoryStyle" :disabled="memoryStyleSaving || !memoryStyleEnabled || !memoryStyle.trim()">
              清除当前风格
            </button>
          </div>
        </div>
        <div v-if="token" class="setting-group" style="margin-top:24px;border-top:1px solid var(--border);padding-top:16px">
          <button class="btn-logout" @click="doLogout">退出登录</button>
        </div>
        <div class="setting-footer-note">
          B站飞翔鲸祝您马年大吉！财源广进！<br>
          WhaleClaw 免费开源！
        </div>
      </div>

      <!-- Skill Detail Modal -->
      <div v-if="skillDetail" class="detail-overlay" @click.self="skillDetail = null">
        <div class="detail-modal">
          <div class="detail-modal-header">
            <h2>🧩 {{ skillDetail.name }}</h2>
            <button class="btn-icon" @click="skillDetail = null" title="关闭">✕</button>
          </div>
          <div class="detail-modal-body markdown-body" v-html="renderMarkdown(skillDetail.raw_markdown || '')"></div>
        </div>
      </div>

      <!-- Tool Detail Modal -->
      <div v-if="toolDetail" class="detail-overlay" @click.self="toolDetail = null">
        <div class="detail-modal">
          <div class="detail-modal-header">
            <h2><code style="font-size:16px">🔧 {{ toolDetail.name }}</code></h2>
            <button class="btn-icon" @click="toolDetail = null" title="关闭">✕</button>
          </div>
          <div class="detail-modal-body">
            <p class="tool-detail-desc">{{ toolDetail.description }}</p>
            <div v-if="toolDetail.parameters && toolDetail.parameters.length" class="tool-detail-params">
              <h3>参数</h3>
              <div v-for="p in toolDetail.parameters" :key="p.name" class="tool-detail-param">
                <div class="tool-detail-param-header">
                  <code>{{ p.name }}</code>
                  <span class="tool-detail-param-type">{{ p.type }}</span>
                  <span v-if="p.required" class="chip chip-accent" style="font-size:11px;padding:1px 6px">必填</span>
                  <span v-else class="chip" style="font-size:11px;padding:1px 6px">可选</span>
                </div>
                <div v-if="p.description" class="tool-detail-param-desc">{{ p.description }}</div>
                <div v-if="p.enum && p.enum.length" class="tool-detail-param-enum">
                  可选值：<code v-for="e in p.enum" :key="e">{{ e }}</code>
                </div>
              </div>
            </div>
            <div v-else class="tool-detail-no-params">此工具无需参数</div>
          </div>
        </div>
      </div>
    </template>
  `,
}).mount('#app');
