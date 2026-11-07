/* ==========================================================================
   Молекула — клиентская часть.
   Весь расчёт выполняет сервис: здесь только ввод, показ ответа и подборка.
   Ни одно число на экране не придумывается в браузере.
   ========================================================================== */
'use strict';

(function () {

  // ───────────────────────── мелкие утилиты ─────────────────────────

  var $ = function (id) { return document.getElementById(id); };

  function esc(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function fmtSI(value) {
    if (value === null || value === undefined || !isFinite(value)) return '—';
    var abs = Math.abs(value);
    var digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
    return Number(value).toLocaleString('ru-RU', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });
  }

  function fmtNum(value, digits) {
    if (value === null || value === undefined || !isFinite(value)) return '—';
    if (digits === undefined) digits = 3;
    return Number(value).toLocaleString('ru-RU', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });
  }

  function fmtInt(value) {
    if (value === null || value === undefined || !isFinite(value)) return '—';
    return Number(value).toLocaleString('ru-RU');
  }

  function pct(value) {
    if (value === null || value === undefined || !isFinite(value)) return '—';
    return fmtNum(Number(value) * 100, 1) + ' %';
  }

  function fmtDate(iso) {
    if (!iso) return '—';
    var date = new Date(iso);
    if (isNaN(date.getTime())) return String(iso);
    return date.toLocaleString('ru-RU', {
      day: '2-digit', month: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  }

  function debounce(fn, wait) {
    var timer = null;
    return function () {
      var args = arguments, self = this;
      clearTimeout(timer);
      timer = setTimeout(function () { fn.apply(self, args); }, wait);
    };
  }

  var toastTimer = null;
  function toast(message) {
    var node = $('toast');
    node.textContent = message;
    node.hidden = false;
    requestAnimationFrame(function () { node.classList.add('is-shown'); });
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      node.classList.remove('is-shown');
      setTimeout(function () { node.hidden = true; }, 220);
    }, 2600);
  }

  // ───────────────────────── перевод сообщений сервиса ─────────────────────────

  var MESSAGES = {
    'SMILES is empty': 'Строка SMILES пустая',
    'SMILES is longer than 2000 characters': 'Строка SMILES длиннее 2000 символов',
    'RDKit cannot parse this string as a molecule': 'RDKit не может разобрать эту строку как молекулу',
    'molecule has no heavy atoms': 'В молекуле нет тяжёлых атомов',
    'nothing left after salt stripping': 'После удаления солевой части ничего не осталось',
    'cannot compute an InChIKey for this molecule': 'Не удалось вычислить InChIKey для этой молекулы',
    'no records supplied': 'Нет строк для расчёта',
    'file is larger than 8 MB': 'Файл больше 8 МБ',
    'cannot decode the file as UTF-8 or CP1251': 'Не удалось прочитать файл: он не в UTF-8 и не в CP1251',
    'the file has no data rows': 'В файле нет строк с данными'
  };

  /** Сообщение сервиса на русском; незнакомый текст показываем как есть. */
  function human(message) {
    if (!message) return '';
    var text = String(message).trim();
    if (MESSAGES[text]) return MESSAGES[text];
    if (text.indexOf('molecule failed standardisation:') === 0) {
      return 'Молекула не прошла стандартизацию: ' + text.slice('molecule failed standardisation:'.length).trim();
    }
    if (text.indexOf('inference failed:') === 0) {
      return 'Расчёт не выполнен: ' + text.slice('inference failed:'.length).trim();
    }
    if (text.indexOf('no SMILES column found.') === 0) {
      return 'В файле нет столбца со SMILES. ' + text.slice('no SMILES column found.'.length).trim()
        .replace('Columns in the file:', 'Столбцы в файле:');
    }
    var rows = text.match(/^(\d+) rows exceeds the limit of (\d+) per request$/);
    if (rows) return 'Строк ' + rows[1] + ', а за один раз можно рассчитать не больше ' + rows[2];
    var file = text.match(/^the file has (\d+) rows, the limit is (\d+)$/);
    if (file) return 'В файле ' + file[1] + ' строк, а допустимо не больше ' + file[2];
    return text;
  }

  // ───────────────────────── запросы к сервису ─────────────────────────

  function ApiError(message, status, raw) {
    this.name = 'ApiError';
    this.message = message;
    this.status = status;
    this.raw = raw;
  }
  ApiError.prototype = Object.create(Error.prototype);

  function request(path, options) {
    return fetch(path, options).then(function (response) {
      return response.text().then(function (text) {
        var payload = null;
        if (text) { try { payload = JSON.parse(text); } catch (e) { payload = null; } }
        if (!response.ok) {
          var detail = payload && (payload.detail || payload.message);
          if (detail && typeof detail !== 'string') detail = JSON.stringify(detail);
          throw new ApiError(human(detail || text || ('HTTP ' + response.status)), response.status, detail || text);
        }
        return payload;
      });
    }, function () {
      throw new ApiError('Сервис недоступен: не удалось соединиться. Проверьте, запущен ли он.', 0, null);
    });
  }

  function postJSON(path, body) {
    return request(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
  }

  /** Понятное объяснение отказа: при 5xx спрашиваем /api/health — там честная причина. */
  function explain(error) {
    if (error && error.status >= 500) {
      return checkHealth().then(function (health) {
        if (health && health.status !== 'ok') {
          return 'Модель не загружена. ' + (health.detail || 'Обучите модель и перезапустите сервис.');
        }
        return error.message || 'Сервис ответил ошибкой.';
      });
    }
    return Promise.resolve((error && error.message) || 'Неизвестная ошибка.');
  }

  // ───────────────────────── состояние ─────────────────────────

  var state = {
    screen: 'analysis',
    health: null,
    single: { loading: false, result: null, error: null, modelVersion: null, datasetVersion: null },
    batch: {
      loading: false,
      error: null,
      rows: [],
      source: '',
      sort: { key: null, dir: 1 },
      selected: {},
      modelVersion: null,
      datasetVersion: null
    },
    data: { loading: false, error: null, model: null, dataset: null, loaded: false },
    worklist: []
  };

  var SCREEN_TITLES = {
    analysis: 'Анализ',
    batch: 'Пакетный расчёт',
    data: 'Данные и модель',
    worklist: 'Подборка'
  };

  // ───────────────────────── тема ─────────────────────────

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }
  function isDark() { return currentTheme() === 'dark'; }

  function setTheme(theme) {
    if (theme !== 'dark') theme = 'light';
    if (theme === currentTheme()) { markThemeButtons(); return; }
    var root = document.documentElement;
    root.setAttribute('data-theme-switching', '');
    root.setAttribute('data-theme', theme);
    try { localStorage.setItem('molecula.theme', theme); } catch (e) { /* приватный режим */ }
    setTimeout(function () { root.removeAttribute('data-theme-switching'); }, 260);
    markThemeButtons();
    redrawStructures();
  }

  function markThemeButtons() {
    var theme = currentTheme();
    Array.prototype.forEach.call(document.querySelectorAll('[data-theme-value]'), function (button) {
      var active = button.getAttribute('data-theme-value') === theme;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
  }

  // ───────────────────────── структуры молекул ─────────────────────────

  var structureCache = {};

  function structureSvg(smiles) {
    var key = (isDark() ? 'd|' : 'l|') + smiles;
    if (structureCache[key]) return Promise.resolve(structureCache[key]);
    return postJSON('/api/structure', { smiles: smiles, dark: isDark() }).then(function (payload) {
      // RDKit отдаёт SVG с XML-прологом — в разметку страницы он не нужен
      var svg = String(payload.svg || '');
      var start = svg.indexOf('<svg');
      structureCache[key] = start > 0 ? svg.slice(start) : svg;
      return structureCache[key];
    });
  }

  function drawStructure(container, smiles) {
    if (!container || !smiles) return;
    container.setAttribute('data-structure', smiles);
    structureSvg(smiles).then(function (svg) {
      if (container.getAttribute('data-structure') !== smiles) return;
      if (!container.isConnected) return;
      container.innerHTML = svg;
    }, function () {
      if (container.getAttribute('data-structure') !== smiles) return;
      container.innerHTML = '<p class="hint">Не удалось нарисовать структуру</p>';
    });
  }

  function redrawStructures() {
    Array.prototype.forEach.call(document.querySelectorAll('[data-structure]'), function (container) {
      drawStructure(container, container.getAttribute('data-structure'));
    });
  }

  // ───────────────────────── состояние сервиса ─────────────────────────

  function checkHealth() {
    return request('/api/health').then(function (payload) {
      state.health = payload;
      renderHealth();
      return payload;
    }, function (error) {
      state.health = { status: 'unreachable', detail: error.message };
      renderHealth();
      return state.health;
    });
  }

  function modelReady() {
    return !!(state.health && state.health.status === 'ok');
  }

  function renderHealth() {
    var dot = $('serviceDot');
    var text = $('serviceText');
    var status = $('serviceStatus');
    if (!state.health) {
      dot.className = 'dot dot--wait';
      text.textContent = 'Проверяем модель…';
      status.title = 'Проверяем состояние модели';
    } else if (state.health.status === 'ok') {
      dot.className = 'dot dot--ok';
      text.textContent = 'Модель: ' + (state.health.model_version || 'загружена');
      status.title = 'Модель загружена: ' + (state.health.model_version || '');
    } else {
      dot.className = 'dot dot--bad';
      text.textContent = 'Модель не загружена';
      status.title = state.health.detail || 'Модель не загружена';
    }
    renderBanners();
    syncRunButtons();
  }

  function renderBanners() {
    ['analysis', 'batch'].forEach(function (name) {
      var screen = $('screen-' + name);
      var stack = screen.querySelector('.stack');
      var banner = screen.querySelector('.model-banner');
      if (modelReady() || !state.health) {
        if (banner) banner.remove();
        return;
      }
      var unreachable = state.health.status === 'unreachable';
      var detail = state.health.detail || 'Файл модели не найден.';
      var text = unreachable
        ? '<strong>Сервис недоступен.</strong> Расчёт невозможен, пока браузер не достучится до сервиса. '
        : '<strong>Модель не загружена.</strong> Расчёт недоступен, пока сервис не обучит и не сохранит модель. ';
      text += 'Ответ сервиса: <span class="mono">' + esc(detail) + '</span><br>' +
        '<button class="btn btn--ghost btn--sm" type="button" data-action="recheck" style="margin-top:12px">Проверить снова</button>';
      if (!banner) {
        banner = document.createElement('div');
        banner.className = 'note note--warn model-banner';
        stack.insertBefore(banner, stack.firstChild);
      }
      // перерисовываем только при изменении текста, иначе фокус на кнопке теряется при каждой проверке
      if (banner.getAttribute('data-detail') !== text) {
        banner.setAttribute('data-detail', text);
        banner.innerHTML = text;
      }
    });
  }

  function syncRunButtons() {
    var ready = modelReady();
    [$('runSingle'), $('runBatch')].forEach(function (button) {
      button.disabled = !ready || (button.id === 'runSingle' ? state.single.loading : state.batch.loading);
      button.title = ready ? '' : 'Модель не загружена — расчёт недоступен';
    });
  }

  // ───────────────────────── переключение экранов ─────────────────────────

  function showScreen(name) {
    if (!SCREEN_TITLES[name]) name = 'analysis';
    state.screen = name;
    Object.keys(SCREEN_TITLES).forEach(function (key) {
      $('screen-' + key).hidden = key !== name;
    });
    Array.prototype.forEach.call(document.querySelectorAll('.nav__item'), function (item) {
      var active = item.getAttribute('data-screen') === name;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-current', active ? 'page' : 'false');
    });
    $('screenTitle').textContent = SCREEN_TITLES[name];
    document.title = SCREEN_TITLES[name] + ' — Молекула';
    try { history.replaceState(null, '', '#' + name); } catch (e) { /* file:// */ }
    closeDrawer();
    if (name === 'data' && !state.data.loaded && !state.data.loading) loadData();
  }

  // ───────────────────────── левое меню ─────────────────────────

  function isNarrow() { return window.matchMedia('(max-width: 900px)').matches; }

  function setCollapsed(collapsed) {
    var root = document.documentElement;
    if (collapsed) root.setAttribute('data-sidebar', 'collapsed');
    else root.removeAttribute('data-sidebar');
    try { localStorage.setItem('molecula.sidebar', collapsed ? 'collapsed' : 'expanded'); } catch (e) { /* ignore */ }
    var toggle = $('sidebarToggle');
    toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    toggle.setAttribute('aria-label', collapsed ? 'Развернуть меню' : 'Свернуть меню');
    toggle.title = collapsed ? 'Развернуть меню' : 'Свернуть меню';
  }

  function openDrawer() {
    $('sidebar').classList.add('is-open');
    var scrim = $('scrim');
    scrim.hidden = false;
    requestAnimationFrame(function () { scrim.classList.add('is-shown'); });
    $('menuBtn').setAttribute('aria-expanded', 'true');
  }

  function closeDrawer() {
    $('sidebar').classList.remove('is-open');
    var scrim = $('scrim');
    scrim.classList.remove('is-shown');
    setTimeout(function () { if (!scrim.classList.contains('is-shown')) scrim.hidden = true; }, 200);
    $('menuBtn').setAttribute('aria-expanded', 'false');
  }

  // ───────────────────────── общие куски разметки ─────────────────────────

  function stateBlock(kind, title, text, actionHtml) {
    var icons = {
      empty: '<circle cx="12" cy="12" r="8.4"></circle><path d="M12 8.4 L12 13 M12 16 L12 16.01"></path>',
      error: '<path d="M12 4.6 L21 19.4 L3 19.4 Z"></path><path d="M12 10 L12 14.2 M12 16.8 L12 16.81"></path>'
    };
    return '<div class="state state--' + kind + '">' +
      '<svg viewBox="0 0 24 24" class="ico state__ico" aria-hidden="true">' + (icons[kind] || icons.empty) + '</svg>' +
      '<p class="state__title">' + esc(title) + '</p>' +
      (text ? '<p class="state__text">' + esc(text) + '</p>' : '') +
      (actionHtml || '') +
      '</div>';
  }

  function loadingBlock(text) {
    return '<div class="state state--loading">' +
      '<span class="spinner" aria-hidden="true"></span>' +
      '<p class="state__title">' + esc(text) + '</p>' +
      '</div>';
  }

  function setPanel(panel, kind, html) {
    panel.setAttribute('data-state', kind);
    panel.innerHTML = '<div class="panel__fade">' + html + '</div>';
    if (kind !== 'loading') panel.style.minHeight = '';
  }

  // Пока идёт пересчёт, панель держит высоту прошлого результата: содержимое меняется, рамка — нет.
  function holdPanelHeight(panel) {
    var height = panel.getBoundingClientRect().height;
    if (height > 0) panel.style.minHeight = Math.round(height) + 'px';
  }

  function copyButton(value) {
    return '<button class="icon-btn" type="button" data-copy="' + esc(value) + '" title="Скопировать" aria-label="Скопировать">' +
      '<svg viewBox="0 0 24 24" class="ico" aria-hidden="true">' +
      '<rect x="9" y="9" width="11" height="11" rx="2.4"></rect>' +
      '<path d="M15 9 L15 6.4 A2.4 2.4 0 0 0 12.6 4 L6.4 4 A2.4 2.4 0 0 0 4 6.4 L4 12.6 A2.4 2.4 0 0 0 6.4 15 L9 15"></path>' +
      '</svg></button>';
  }

  function monoRow(value) {
    if (!value) return '<span class="hint">—</span>';
    return '<span class="copy-row"><span class="mono">' + esc(value) + '</span>' + copyButton(value) + '</span>';
  }

  function applicabilityBadge(result) {
    if (result.applicability === 'in_domain') {
      return '<span class="badge badge--in">В области обучающих данных</span>';
    }
    if (result.applicability === 'out_of_domain') {
      return '<span class="badge badge--out">За пределами обучающих данных</span>';
    }
    return '';
  }

  function applicabilityText(result) {
    var similarity = fmtNum(result.nearest_similarity, 2);
    var head = result.applicability === 'out_of_domain'
      ? 'Молекула далека от того, на чём училась модель: ближайшее сходство по Танимото — ' + similarity + ' (шкала 0–1). Для неё прогноз является экстраполяцией.'
      : 'Молекула похожа на обучающие данные: ближайшее сходство по Танимото — ' + similarity + ' (шкала 0–1).';
    var tail = result.reliability_assessed === false
      ? ' Надёжность этого прогноза не оценена: на проверочной выборке не удалось показать, что сходство отделяет точные прогнозы от неточных.'
      : ' Сходство описывает похожесть на обучающие данные.';
    return head + tail + ' Это не вероятность того, что прогноз верен.';
  }

  function doiLinks(value) {
    if (!value) return '<span class="hint">—</span>';
    var parts = String(value).split(/[;,\s]+/).filter(function (part) { return part; });
    if (!parts.length) return '<span class="hint">—</span>';
    return parts.map(function (doi) {
      var clean = doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, '');
      return '<a href="https://doi.org/' + encodeURI(clean) + '" target="_blank" rel="noopener noreferrer">' + esc(clean) + '</a>';
    }).join(', ');
  }

  // ───────────────────────── экран «Анализ» ─────────────────────────

  function setFieldError(inputId, messageId, message) {
    var input = $(inputId);
    var holder = $(messageId);
    holder.textContent = message || '';
    if (message) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
  }

  function singleSkeleton() {
    return '<div class="result-grid">' +
      '<div class="skeleton sk-block" style="height:240px"></div>' +
      '<div class="readout">' +
      '<div class="skeleton sk-line" style="width:140px"></div>' +
      '<div class="skeleton sk-block" style="height:44px;width:60%"></div>' +
      '<div class="skeleton sk-line" style="width:80%"></div>' +
      '<div class="skeleton sk-line" style="width:70%"></div>' +
      '<div class="skeleton sk-block" style="height:64px"></div>' +
      '</div></div>' +
      '<div class="divider"></div>' +
      '<div class="skeleton sk-line" style="width:180px;margin-bottom:16px"></div>' +
      '<div class="skeleton sk-block" style="height:120px"></div>';
  }

  function renderSingle() {
    var panel = $('singleResult');
    var single = state.single;

    if (single.loading) {
      holdPanelHeight(panel);
      panel.setAttribute('data-state', 'loading');
      panel.innerHTML = '<div class="panel__fade">' + singleSkeleton() + '</div>';
      return;
    }
    if (single.error) {
      setPanel(panel, 'error', stateBlock('error', 'Расчёт не выполнен', single.error));
      return;
    }
    if (!single.result) {
      setPanel(panel, 'empty', stateBlock('empty', 'Пока нечего показать',
        'Введите SMILES и нажмите «Рассчитать» — сервис вернёт прогноз индекса селективности SI = CC50/IC50.',
        '<button class="btn btn--ghost btn--sm" type="button" data-example="NC1=NC(=O)N(C[C@@H](CO)OCP(=O)(O)O)C=C1">Подставить пример</button>'));
      return;
    }

    var result = single.result;

    if (result.status === 'invalid') {
      setPanel(panel, 'error', stateBlock('error', 'Молекула не принята',
        human(result.error)) +
        '<p class="hint" style="text-align:center">Ответ сервиса: <span class="mono">' + esc(result.error || '') + '</span></p>');
      return;
    }

    var neighbours = result.neighbours || [];
    var html = '';

    html += '<div class="result-grid">';
    html += '<div class="structure" data-structure="' + esc(result.canonical_smiles || result.input_smiles || '') + '">' +
      '<span class="spinner" aria-hidden="true"></span></div>';
    html += '<div class="readout">';
    html += '<div><p class="readout__label">Прогноз модели — индекс селективности SI</p>' +
      '<p class="readout__value">' + esc(fmtSI(result.prediction_si)) +
      '<span class="readout__unit">SI = CC50 / IC50, безразмерная величина</span></p></div>';
    html += '<div class="badges">' + applicabilityBadge(result) +
      '<span class="badge">log10(SI) = ' + esc(fmtNum(result.prediction_log10_si, 3)) + '</span>' +
      '<span class="badge">Сходство по Танимото ' + esc(fmtNum(result.nearest_similarity, 2)) + '</span>' +
      (result.reliability_assessed === false ? '<span class="badge badge--bad">Надёжность не оценена</span>' : '') +
      '</div>';
    html += '<p class="note">' + esc(applicabilityText(result)) + '</p>';
    html += '<div class="field__row">' +
      '<button class="btn btn--primary btn--sm" type="button" data-action="single-to-worklist">В подборку</button>' +
      '<span class="hint">Версия модели: ' + esc(state.single.modelVersion || '—') + '</span></div>';
    html += '</div></div>';

    html += '<div class="divider"></div>';
    html += '<h2 class="section-title">Паспорт результата <span class="section-sub">что именно посчитано и на чём</span></h2>';
    html += '<dl class="kv" style="margin-top:16px">';
    html += '<dt>Введённый SMILES</dt><dd>' + monoRow(result.input_smiles) + '</dd>';
    html += '<dt>Канонический SMILES</dt><dd>' + monoRow(result.canonical_smiles) + '</dd>';
    html += '<dt>InChIKey</dt><dd>' + monoRow(result.inchikey) + '</dd>';
    html += '<dt>Версия модели</dt><dd class="mono">' + esc(state.single.modelVersion || '—') + '</dd>';
    html += '<dt>Версия набора данных</dt><dd class="mono">' + esc(state.single.datasetVersion || '—') + '</dd>';
    html += '<dt>Дата расчёта</dt><dd title="' + esc(result.computed_at || '') + '">' + esc(fmtDate(result.computed_at)) + '</dd>';
    html += '</dl>';

    html += '<div class="divider"></div>';
    html += '<h2 class="section-title">Ближайшие молекулы из обучающих данных ' +
      '<span class="section-sub">SI здесь — измеренные значения из публикаций, не прогноз</span></h2>';

    if (!neighbours.length) {
      html += '<p class="note" style="margin-top:16px">Сервис не вернул ближайших примеров.</p>';
    } else {
      html += '<div class="table-wrap" style="margin-top:16px"><table class="t"><thead><tr>' +
        '<th><span class="th-plain">Структура (SMILES)</span></th>' +
        '<th><span class="th-plain">InChIKey</span></th>' +
        '<th><span class="th-plain">Измеренный SI</span></th>' +
        '<th><span class="th-plain">Сходство</span></th>' +
        '<th><span class="th-plain">Публикации (DOI)</span></th>' +
        '</tr></thead><tbody>';
      neighbours.forEach(function (neighbour) {
        html += '<tr>' +
          '<td class="cell-smiles">' + esc(neighbour.smiles) + '</td>' +
          '<td class="mono cell-key">' + esc(neighbour.inchikey) + '</td>' +
          '<td class="num">' + esc(fmtSI(neighbour.si)) + '</td>' +
          '<td class="num">' + esc(fmtNum(neighbour.similarity, 2)) + '</td>' +
          '<td>' + doiLinks(neighbour.dois) +
          (neighbour.documents ? '<div class="hint mono">' + esc(neighbour.documents) + '</div>' : '') +
          '</td></tr>';
      });
      html += '</tbody></table></div>';
      html += '<p class="hint" style="margin-top:12px">Измеренные значения взяты из ChEMBL и приведены как есть. ' +
        'Крупное число выше — прогноз модели для введённой молекулы.</p>';
    }

    setPanel(panel, 'ok', html);
    var structure = panel.querySelector('[data-structure]');
    if (structure) drawStructure(structure, structure.getAttribute('data-structure'));
  }

  function runSingle() {
    var smiles = $('smilesInput').value.trim();
    setFieldError('smilesInput', 'smilesError', '');
    if (!smiles) {
      setFieldError('smilesInput', 'smilesError', 'Введите SMILES молекулы.');
      $('smilesInput').focus();
      return;
    }
    if (!modelReady()) { checkHealth(); return; }

    state.single.loading = true;
    state.single.error = null;
    state.single.result = null;
    syncRunButtons();
    renderSingle();

    postJSON('/predict', {
      records: [{ id: 'single', smiles: smiles }],
      include_structure: false,
      include_neighbours: true
    }).then(function (payload) {
      state.single.loading = false;
      state.single.modelVersion = payload.model_version;
      state.single.datasetVersion = payload.dataset_version;
      var result = (payload.results && payload.results[0]) || null;
      state.single.result = result;
      if (result && result.status === 'invalid') {
        setFieldError('smilesInput', 'smilesError', human(result.error));
      }
      syncRunButtons();
      renderSingle();
    }, function (error) {
      state.single.loading = false;
      explain(error).then(function (message) {
        state.single.error = message;
        syncRunButtons();
        renderSingle();
      });
    });
  }

  // ───────────────────────── экран «Пакетный расчёт» ─────────────────────────

  var COLUMNS = [
    { key: 'id', title: 'Идентификатор', sortable: true },
    { key: 'smiles', title: 'SMILES', sortable: true },
    { key: 'status', title: 'Проверка', sortable: true },
    { key: 'si', title: 'Прогноз SI', sortable: true },
    { key: 'applicability', title: 'Применимость', sortable: true },
    { key: 'error', title: 'Сообщение об ошибке', sortable: true }
  ];

  function statusLabel(row) {
    return row.status === 'ok' ? 'принято' : 'не принято';
  }

  function applicabilityLabel(row) {
    if (row.applicability === 'in_domain') return 'в области данных';
    if (row.applicability === 'out_of_domain') return 'за пределами данных';
    return '—';
  }

  function cellValue(row, key) {
    switch (key) {
      case 'id': return String(row.id);
      case 'smiles': return row.input_smiles || '';
      case 'status': return statusLabel(row);
      case 'si': return row.status === 'ok' ? fmtSI(row.prediction_si) : '—';
      case 'applicability': return row.status === 'ok' ? applicabilityLabel(row) : '—';
      case 'error': return row.status === 'ok' ? '' : human(row.error);
      default: return '';
    }
  }

  function sortValue(row, key) {
    if (key === 'si') return row.status === 'ok' && row.prediction_si !== null ? row.prediction_si : null;
    return cellValue(row, key).toLowerCase();
  }

  function sortedRows() {
    var rows = state.batch.rows.slice();
    var sort = state.batch.sort;
    if (!sort.key) return rows;
    var collator = new Intl.Collator('ru', { numeric: true, sensitivity: 'base' });
    rows.sort(function (a, b) {
      var left = sortValue(a, sort.key);
      var right = sortValue(b, sort.key);
      if (left === null && right === null) return a.__index - b.__index;
      if (left === null) return 1;            // строки без числа — всегда внизу
      if (right === null) return -1;
      var cmp = typeof left === 'number' ? left - right : collator.compare(left, right);
      if (cmp === 0) return a.__index - b.__index;
      return cmp * sort.dir;
    });
    return rows;
  }

  function selectedRows() {
    return state.batch.rows.filter(function (row) { return state.batch.selected[row.__key]; });
  }

  function renderBatch() {
    var panel = $('batchPanel');
    var batch = state.batch;

    if (batch.loading) {
      holdPanelHeight(panel);
      panel.setAttribute('data-state', 'loading');
      panel.innerHTML = '<div class="panel__fade">' + loadingBlock(batch.phase === 'file'
        ? 'Читаем файл…'
        : 'Считаем… строк в работе: ' + fmtInt(batch.pending || 0)) + '</div>';
      return;
    }
    if (batch.error) {
      setPanel(panel, 'error', stateBlock('error', batch.errorTitle || 'Расчёт не выполнен', batch.error));
      return;
    }
    if (!batch.rows.length) {
      setPanel(panel, 'empty', stateBlock('empty', 'Таблица пуста',
        'Загрузите CSV или вставьте список SMILES и нажмите «Рассчитать». Строка с ошибкой не останавливает расчёт остальных.'));
      return;
    }

    var rows = sortedRows();
    var chosen = selectedRows();
    var ok = batch.rows.filter(function (row) { return row.status === 'ok'; }).length;
    var bad = batch.rows.length - ok;

    var html = '';
    html += '<div class="toolbar">';
    html += '<span class="toolbar__count">Строк: ' + fmtInt(batch.rows.length) +
      ' · принято: ' + fmtInt(ok) + ' · не принято: ' + fmtInt(bad) +
      (chosen.length ? ' · выбрано: ' + fmtInt(chosen.length) : '') + '</span>';
    html += '<span class="toolbar__spacer"></span>';
    html += '<button class="btn btn--ghost btn--sm" type="button" data-action="compare"' +
      (chosen.length < 1 ? ' disabled' : '') + '>Сравнить</button>';
    html += '<button class="btn btn--ghost btn--sm" type="button" data-action="batch-to-worklist"' +
      (chosen.length < 1 ? ' disabled' : '') + '>В подборку</button>';
    html += '<button class="btn btn--primary btn--sm" type="button" data-action="export-batch">Выгрузить CSV</button>';
    html += '</div>';

    html += '<p class="hint" style="margin-bottom:12px">Прогноз SI — результат модели. Версия модели: ' +
      esc(batch.modelVersion || '—') + ', набор данных: ' + esc(batch.datasetVersion || '—') + '.</p>';

    html += '<div class="table-wrap"><table class="t"><thead><tr>';
    var allSelected = batch.rows.length > 0 && chosen.length === batch.rows.length;
    html += '<th class="cell-check"><span class="th-plain">' +
      '<input type="checkbox" data-action="select-all" aria-label="Выбрать все строки"' +
      (allSelected ? ' checked' : '') + '></span></th>';
    COLUMNS.forEach(function (column) {
      var active = batch.sort.key === column.key;
      var ariaSort = active ? (batch.sort.dir === 1 ? 'ascending' : 'descending') : null;
      html += '<th' + (ariaSort ? ' aria-sort="' + ariaSort + '"' : '') + '>' +
        '<button class="th-sort" type="button" data-sort="' + column.key + '">' +
        esc(column.title) +
        '<span class="th-sort__arrow" aria-hidden="true">' + (active && batch.sort.dir === -1 ? '▼' : '▲') + '</span>' +
        '</button></th>';
    });
    html += '</tr></thead><tbody>';

    rows.forEach(function (row) {
      var selected = !!batch.selected[row.__key];
      html += '<tr' + (selected ? ' class="is-selected"' : '') + '>';
      html += '<td class="cell-check"><input type="checkbox" data-row="' + esc(row.__key) + '"' +
        (selected ? ' checked' : '') + ' aria-label="Выбрать строку ' + esc(row.id) + '"></td>';
      html += '<td class="cell-id">' + esc(cellValue(row, 'id')) + '</td>';
      html += '<td class="cell-smiles">' + esc(cellValue(row, 'smiles')) + '</td>';
      html += '<td>' + (row.status === 'ok'
        ? '<span class="badge badge--in">принято</span>'
        : '<span class="badge badge--bad">не принято</span>') + '</td>';
      html += '<td class="num">' + esc(cellValue(row, 'si')) + '</td>';
      html += '<td>' + (row.status === 'ok' && row.applicability
        ? (row.applicability === 'in_domain'
          ? '<span class="badge badge--in">в области данных</span>'
          : '<span class="badge badge--out">за пределами данных</span>')
        : '<span class="hint">—</span>') + '</td>';
      html += '<td class="cell-error">' + esc(cellValue(row, 'error')) + '</td>';
      html += '</tr>';
    });

    html += '</tbody></table></div>';
    html += '<p class="table-hint">Таблица прокручивается вбок — прогноз SI и применимость правее.</p>';
    html += '<p class="hint" style="margin-top:12px">Выгрузка повторяет таблицу: те же строки, тот же порядок и те же числа.</p>';

    setPanel(panel, 'ok', html);
  }

  function parsePastedSmiles(text) {
    return text.split(/\r?\n/)
      .map(function (line) { return line.trim(); })
      .filter(function (line) { return line.length > 0; })
      .map(function (line, index) {
        // «id,SMILES» или «id<TAB>SMILES» — если в строке два поля, первое считаем идентификатором
        var parts = line.split(/[\t;,]/).map(function (part) { return part.trim(); }).filter(Boolean);
        if (parts.length >= 2) return { id: parts[0], smiles: parts.slice(1).join('') };
        return { id: 'строка-' + (index + 1), smiles: line };
      });
  }

  function runBatch(records, source) {
    // расчёт мог не начаться — панель не должна остаться в состоянии загрузки
    if (!records || !records.length) {
      state.batch.loading = false;
      setFieldError('batchText', 'batchError', 'Добавьте хотя бы одну строку со SMILES.');
      renderBatch();
      return;
    }
    if (!modelReady()) {
      state.batch.loading = false;
      $('batchSource').textContent = source || '';
      renderBatch();
      checkHealth();
      return;
    }
    setFieldError('batchText', 'batchError', '');

    state.batch.loading = true;
    state.batch.phase = 'predict';
    state.batch.pending = records.length;
    state.batch.error = null;
    state.batch.errorTitle = null;
    state.batch.selected = {};
    state.batch.source = source || '';
    $('batchSource').textContent = source || '';
    syncRunButtons();
    renderBatch();

    postJSON('/predict', {
      records: records,
      include_structure: false,
      include_neighbours: true
    }).then(function (payload) {
      state.batch.loading = false;
      state.batch.modelVersion = payload.model_version;
      state.batch.datasetVersion = payload.dataset_version;
      state.batch.rows = (payload.results || []).map(function (row, index) {
        row.__index = index;
        row.__key = 'r' + index;
        if (!row.input_smiles && records[index]) row.input_smiles = records[index].smiles;
        return row;
      });
      state.batch.sort = { key: null, dir: 1 };
      syncRunButtons();
      renderBatch();
    }, function (error) {
      state.batch.loading = false;
      explain(error).then(function (message) {
        state.batch.error = message;
        syncRunButtons();
        renderBatch();
      });
    });
  }

  function uploadCsv(file) {
    if (!file) return;
    setFieldError('batchText', 'batchError', '');
    state.batch.loading = true;
    state.batch.phase = 'file';
    state.batch.pending = 0;
    state.batch.error = null;
    state.batch.errorTitle = null;
    renderBatch();
    $('batchSource').textContent = 'Читаем файл ' + file.name + '…';

    var form = new FormData();
    form.append('file', file, file.name);
    request('/api/parse-csv', { method: 'POST', body: form }).then(function (payload) {
      state.batch.loading = false;
      var note = 'Файл ' + payload.filename + ': строк ' + payload.records.length +
        ', SMILES из столбца «' + payload.smiles_column + '»' +
        (payload.id_column ? ', идентификатор из столбца «' + payload.id_column + '»' : ', идентификаторы проставлены по порядку строк');
      runBatch(payload.records, note);
    }, function (error) {
      state.batch.loading = false;
      $('batchSource').textContent = '';
      setFieldError('batchText', 'batchError', error.message);
      state.batch.error = error.message;
      state.batch.errorTitle = 'Файл не прочитан';
      renderBatch();
    });
  }

  // ───────────────────────── выгрузка CSV ─────────────────────────

  function csvCell(value) {
    var text = value === null || value === undefined ? '' : String(value);
    if (/[";\n\r]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
    return text;
  }

  function downloadCsv(filename, header, rows) {
    var lines = [header.map(csvCell).join(';')];
    rows.forEach(function (row) { lines.push(row.map(csvCell).join(';')); });
    var blob = new Blob(['﻿' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function exportBatch() {
    var rows = sortedRows();
    downloadCsv(
      'molecula-prognoz-' + new Date().toISOString().slice(0, 10) + '.csv',
      COLUMNS.map(function (column) { return column.title; }),
      rows.map(function (row) {
        return COLUMNS.map(function (column) { return cellValue(row, column.key); });
      })
    );
    toast('Файл выгружен: ' + rows.length + ' строк');
  }

  // ───────────────────────── подборка ─────────────────────────

  var WORKLIST_KEY = 'molecula.worklist.v1';

  function loadWorklist() {
    try {
      var raw = localStorage.getItem(WORKLIST_KEY);
      state.worklist = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(state.worklist)) state.worklist = [];
    } catch (e) {
      state.worklist = [];
    }
  }

  function saveWorklist() {
    try { localStorage.setItem(WORKLIST_KEY, JSON.stringify(state.worklist)); } catch (e) { /* приватный режим */ }
  }

  function worklistKeyOf(entry) {
    return entry.inchikey || entry.canonical_smiles || entry.input_smiles || String(entry.id);
  }

  function addToWorklist(results, modelVersion, datasetVersion) {
    var added = 0, skipped = 0;
    results.forEach(function (result) {
      if (!result || result.status !== 'ok') { skipped++; return; }
      var key = worklistKeyOf(result);
      var exists = state.worklist.some(function (item) { return item.key === key; });
      if (exists) { skipped++; return; }
      state.worklist.push({
        key: key,
        id: String(result.id),
        input_smiles: result.input_smiles || '',
        canonical_smiles: result.canonical_smiles || '',
        inchikey: result.inchikey || '',
        prediction_si: result.prediction_si,
        prediction_log10_si: result.prediction_log10_si,
        applicability: result.applicability || '',
        nearest_similarity: result.nearest_similarity,
        reliability_assessed: result.reliability_assessed,
        model_version: modelVersion || '',
        dataset_version: datasetVersion || '',
        computed_at: result.computed_at || '',
        note: '',
        added_at: new Date().toISOString()
      });
      added++;
    });
    saveWorklist();
    renderWorklistCount();
    renderWorklist();
    if (added && skipped) toast('Добавлено: ' + added + ', пропущено: ' + skipped);
    else if (added) toast(added === 1 ? 'Добавлено в подборку' : 'Добавлено в подборку: ' + added);
    else toast('Нечего добавлять: эти молекулы уже в подборке или не приняты');
  }

  function renderWorklistCount() {
    var badge = $('worklistCount');
    badge.textContent = String(state.worklist.length);
    badge.hidden = state.worklist.length === 0;
  }

  function renderWorklist() {
    var panel = $('worklistPanel');
    if (!state.worklist.length) {
      setPanel(panel, 'empty', stateBlock('empty', 'Подборка пуста',
        'Отметьте кандидатов в пакетном расчёте или нажмите «В подборку» на экране анализа. Подборка хранится в этом браузере.',
        '<button class="btn btn--ghost btn--sm" type="button" data-screen-link="analysis">Перейти к анализу</button>'));
      return;
    }

    var html = '';
    html += '<div class="toolbar">';
    html += '<span class="toolbar__count">Молекул в подборке: ' + fmtInt(state.worklist.length) + '</span>';
    html += '<span class="toolbar__spacer"></span>';
    html += '<button class="btn btn--ghost btn--sm" type="button" data-action="worklist-clear">Очистить</button>';
    html += '<button class="btn btn--primary btn--sm" type="button" data-action="export-worklist">Выгрузить CSV</button>';
    html += '</div>';
    html += '<p class="hint" style="margin-bottom:16px">Все значения SI здесь — прогноз модели. Подборка и заметки хранятся только в этом браузере.</p>';
    html += '<ul class="wl-list">';

    state.worklist.forEach(function (item) {
      html += '<li class="wl-item">';
      html += '<div class="wl-item__head">' +
        '<span class="wl-item__id">' + esc(item.id) + '</span>' +
        '<span class="wl-item__si">SI ' + esc(fmtSI(item.prediction_si)) + '</span>' +
        '<span class="hint">прогноз модели</span>' +
        (item.applicability === 'out_of_domain'
          ? '<span class="badge badge--out">за пределами данных</span>'
          : item.applicability === 'in_domain'
            ? '<span class="badge badge--in">в области данных</span>' : '') +
        (item.reliability_assessed === false ? '<span class="badge badge--bad">надёжность не оценена</span>' : '') +
        '</div>';
      html += '<button class="btn btn--ghost btn--sm" type="button" data-remove="' + esc(item.key) + '">Убрать</button>';
      html += '<div class="wl-item__body">';
      html += '<p class="mono">' + esc(item.canonical_smiles || item.input_smiles) + '</p>';
      html += '<p class="hint">InChIKey: ' + esc(item.inchikey || '—') + ' · модель: ' + esc(item.model_version || '—') +
        ' · расчёт: ' + esc(fmtDate(item.computed_at)) + '</p>';
      html += '<label class="field__label" for="note-' + esc(item.key) + '">Заметка</label>';
      html += '<textarea class="input" id="note-' + esc(item.key) + '" rows="2" data-note="' + esc(item.key) +
        '" placeholder="Зачем эта молекула в подборке">' + esc(item.note || '') + '</textarea>';
      html += '</div>';
      html += '</li>';
    });

    html += '</ul>';
    setPanel(panel, 'ok', html);
  }

  function exportWorklist() {
    downloadCsv(
      'molecula-podborka-' + new Date().toISOString().slice(0, 10) + '.csv',
      ['Идентификатор', 'SMILES', 'Канонический SMILES', 'InChIKey', 'Прогноз SI', 'log10(SI)',
        'Применимость', 'Сходство по Танимото', 'Надёжность оценена', 'Версия модели', 'Версия данных', 'Дата расчёта', 'Заметка'],
      state.worklist.map(function (item) {
        return [
          item.id,
          item.input_smiles,
          item.canonical_smiles,
          item.inchikey,
          fmtSI(item.prediction_si),
          fmtNum(item.prediction_log10_si, 3),
          item.applicability === 'in_domain' ? 'в области данных' : item.applicability === 'out_of_domain' ? 'за пределами данных' : '—',
          fmtNum(item.nearest_similarity, 2),
          item.reliability_assessed === false ? 'нет' : item.reliability_assessed === true ? 'да' : '—',
          item.model_version,
          item.dataset_version,
          fmtDate(item.computed_at),
          item.note || ''
        ];
      })
    );
    toast('Файл выгружен: ' + state.worklist.length + ' строк');
  }

  // ───────────────────────── сравнение ─────────────────────────

  var compareRows = [];

  function openCompare(rows) {
    compareRows = rows;
    var body = $('compareBody');
    var html = '<div class="compare">';
    rows.forEach(function (row) {
      html += '<div class="compare__col">';
      html += '<p class="compare__id">' + esc(row.id) + '</p>';
      if (row.status === 'ok') {
        html += '<div class="structure structure--small" data-structure="' +
          esc(row.canonical_smiles || row.input_smiles || '') + '"><span class="spinner" aria-hidden="true"></span></div>';
      } else {
        html += '<div class="structure structure--small"><span class="hint">Молекула не принята</span></div>';
      }
      html += '<dl class="compare__rows">';
      if (row.status === 'ok') {
        html += '<div class="compare__row"><dt class="compare__key">Прогноз SI (модель), безразмерная величина</dt>' +
          '<dd class="compare__val">' + esc(fmtSI(row.prediction_si)) + '</dd></div>';
        html += '<div class="compare__row"><dt class="compare__key">log10(SI), прогноз</dt>' +
          '<dd class="compare__val">' + esc(fmtNum(row.prediction_log10_si, 3)) + '</dd></div>';
        html += '<div class="compare__row"><dt class="compare__key">Применимость</dt>' +
          '<dd class="compare__val">' + esc(applicabilityLabel(row)) + '</dd></div>';
        html += '<div class="compare__row"><dt class="compare__key">Сходство по Танимото с обучающими данными (0–1)</dt>' +
          '<dd class="compare__val">' + esc(fmtNum(row.nearest_similarity, 2)) + '</dd></div>';
        var nearest = (row.neighbours || [])[0];
        html += '<div class="compare__row"><dt class="compare__key">Ближайшая молекула: измеренный SI (публикация)</dt>' +
          '<dd class="compare__val">' + (nearest ? esc(fmtSI(nearest.si)) : '—') + '</dd></div>';
        html += '<div class="compare__row"><dt class="compare__key">Надёжность прогноза</dt>' +
          '<dd class="compare__val">' + (row.reliability_assessed === false ? 'не оценена' : 'оценена') + '</dd></div>';
        html += '<div class="compare__row"><dt class="compare__key">InChIKey</dt>' +
          '<dd class="compare__val mono">' + esc(row.inchikey || '—') + '</dd></div>';
        html += '<div class="compare__row"><dt class="compare__key">Канонический SMILES</dt>' +
          '<dd class="compare__val mono">' + esc(row.canonical_smiles || '—') + '</dd></div>';
      } else {
        html += '<div class="compare__row"><dt class="compare__key">Состояние</dt>' +
          '<dd class="compare__val">Не принято</dd></div>';
        html += '<div class="compare__row"><dt class="compare__key">Сообщение</dt>' +
          '<dd class="compare__val">' + esc(human(row.error)) + '</dd></div>';
        html += '<div class="compare__row"><dt class="compare__key">Введённый SMILES</dt>' +
          '<dd class="compare__val mono">' + esc(row.input_smiles || '—') + '</dd></div>';
      }
      html += '</dl></div>';
    });
    html += '</div>';
    body.innerHTML = html;

    var modal = $('compareModal');
    if (typeof modal.showModal === 'function') modal.showModal();
    else modal.setAttribute('open', '');

    Array.prototype.forEach.call(body.querySelectorAll('[data-structure]'), function (container) {
      drawStructure(container, container.getAttribute('data-structure'));
    });
  }

  function closeCompare() {
    var modal = $('compareModal');
    if (typeof modal.close === 'function' && modal.open) modal.close();
    else modal.removeAttribute('open');
  }

  // ───────────────────────── экран «Данные и модель» ─────────────────────────

  var DOMAIN_LABELS = {
    indicator: 'Показатель',
    threshold: 'Порог',
    n_points: 'Молекул в проверке',
    n_validation: 'Молекул в проверке',
    share_below_threshold: 'Доля ниже порога',
    mean_similarity: 'Среднее сходство',
    spearman_similarity_vs_abs_log_error: 'Спирмен: сходство против ошибки',
    spearman_similarity_vs_abs_error: 'Спирмен: сходство против ошибки',
    spearman_p_value: 'p-значение',
    median_abs_log10_error_inside: 'Медиана |ошибки| log10 внутри области',
    median_abs_log10_error_outside: 'Медиана |ошибки| log10 за пределами',
    rmse_inside_domain: 'RMSE внутри области',
    rmse_outside_domain: 'RMSE за пределами области',
    separates_error: 'Показатель отделяет точные прогнозы от неточных',
    note: 'Примечание'
  };

  var CV_LABELS = {
    molecule: 'Случайное по молекулам (как разделено соревнование)',
    scaffold: 'Целые каркасы Мурко в одной части',
    cluster: 'Целые химические семейства',
    publication: 'Целые источники'
  };

  var COLLECTION_LABELS = {
    dataset_version: 'Версия набора',
    collected_at: 'Собран',
    source: 'Источник',
    vacv_assays: 'Экспериментов по вирусу осповакцины',
    publications_scanned: 'Просмотрено публикаций',
    activities_scanned: 'Просмотрено измерений',
    normalised_measurements: 'Приведено к единым единицам',
    si_pairs: 'Пар IC50/CC50',
    molecules: 'Молекул в наборе',
    excluded_records: 'Исключено записей',
    cache_hits: 'Ответов из кеша',
    cache_misses: 'Запросов к ChEMBL'
  };

  function valueText(key, value) {
    if (value === null || value === undefined) return '—';
    if (typeof value === 'boolean') return value ? 'да' : 'нет';
    if (typeof value === 'number') {
      if (Number.isInteger(value) && Math.abs(value) >= 1 && key !== 'threshold') return fmtInt(value);
      return fmtNum(value, 3);
    }
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  function kvBlock(source, labels, order) {
    if (!source) return '<p class="hint">Сервис не вернул эти данные.</p>';
    var keys = (order || Object.keys(labels)).filter(function (key) {
      return Object.prototype.hasOwnProperty.call(source, key);
    });
    Object.keys(source).forEach(function (key) {
      if (keys.indexOf(key) === -1 && typeof source[key] !== 'object') keys.push(key);
    });
    if (!keys.length) return '<p class="hint">Сервис не вернул эти данные.</p>';
    var html = '<dl class="kv">';
    keys.forEach(function (key) {
      html += '<dt>' + esc(labels[key] || key) + '</dt><dd>' + esc(valueText(key, source[key])) + '</dd>';
    });
    return html + '</dl>';
  }

  function loadData() {
    state.data.loading = true;
    state.data.error = null;
    renderData();
    Promise.all([
      request('/api/model').catch(function (error) { return { __error: error }; }),
      request('/api/dataset').catch(function (error) { return { __error: error }; })
    ]).then(function (parts) {
      var failed = parts.find(function (part) { return part && part.__error; });
      if (failed) {
        return explain(failed.__error).then(function (message) {
          state.data.loading = false;
          state.data.error = message;
          renderData();
        });
      }
      state.data.loading = false;
      state.data.loaded = true;
      state.data.model = parts[0];
      state.data.dataset = parts[1];
      renderData();
    });
  }

  function renderData() {
    var panel = $('dataPanel');

    if (state.data.loading) {
      panel.setAttribute('data-state', 'loading');
      panel.innerHTML = '<div class="panel__fade">' + loadingBlock('Читаем описание модели и набора данных…') + '</div>';
      return;
    }
    if (state.data.error) {
      setPanel(panel, 'error', stateBlock('error', 'Описание недоступно', state.data.error) +
        '<div class="state" style="min-height:0;padding-top:0">' +
        '<button class="btn btn--ghost btn--sm" type="button" data-action="reload-data">Попробовать снова</button></div>');
      return;
    }
    if (!state.data.model || !state.data.dataset) {
      setPanel(panel, 'empty', stateBlock('empty', 'Описание модели и данных',
        'Здесь появятся версия набора данных, проверочные метрики, список публикаций и ограничения — как только сервис их отдаст.'));
      return;
    }

    var model = state.data.model;
    var dataset = state.data.dataset;
    var metrics = dataset.metrics || model.metrics || {};
    var domain = metrics.applicability_domain || {};
    var crossValidation = metrics.cross_validation || {};
    var versusZero = metrics.versus_constant_zero || {};
    var byMolecule = metrics.versus_constant_zero_by_molecule || {};
    var concentration = (dataset.error_analysis || {}).metric_concentration || {};

    var html = '';

    html += '<h2 class="section-title">Модель</h2>';
    html += '<div style="margin-top:16px">' + kvBlock({
      model_version: model.model_version,
      dataset_version: model.dataset_version,
      trained_at: fmtDate(model.trained_at),
      n_training_rows: (model.config || {}).n_training_rows || model.n_training_molecules,
      n_reference_molecules: model.n_training_molecules,
      selected: metrics.selected_candidate || metrics.selected,
      domain_threshold: model.domain_threshold,
      target: (model.feature_contract || {}).target,
      fingerprint: (model.feature_contract || {}).fingerprint,
      seed: (model.config || {}).seed
    }, {
      model_version: 'Версия модели',
      dataset_version: 'Версия набора данных',
      trained_at: 'Обучена',
      n_training_rows: 'Молекул в обучении',
      n_reference_molecules: 'Молекул в справочнике (включая обучающие)',
      selected: 'Выбранный алгоритм',
      domain_threshold: 'Порог области применимости',
      target: 'Целевая величина',
      fingerprint: 'Признаки',
      seed: 'Зерно случайности'
    }) + '</div>';

    if (metrics.selection) {
      html += '<p class="note" style="margin-top:16px">' + esc(metrics.selection) + '</p>';
    }

    if (concentration.share_from_top) {
      html += '<div class="divider"></div>';
      html += '<h2 class="section-title">Из чего состоит метрика</h2>';
      html += '<p class="note note--warn" style="margin-top:16px">' +
        'Официальная метрика — MSE на сыром SI, поэтому вклад молекулы растёт как квадрат её значения. ' +
        'Самая большая молекула обучающего набора даёт ' + pct(concentration.share_from_top['1']) +
        ' всей суммы квадратов, пять самых больших — ' + pct(concentration.share_from_top['5']) + '. ' +
        'Средняя ошибка по выборке почти ничего не говорит о том, насколько хорош прогноз для обычной молекулы.' +
        '</p>';
    }

    var schemes = Object.keys(crossValidation);
    if (schemes.length) {
      html += '<div class="divider"></div>';
      html += '<h2 class="section-title">Перекрёстная проверка ' +
        '<span class="section-sub">одни и те же фолды для всех подходов, метрика соревнования</span></h2>';
      html += '<div class="table-wrap" style="margin-top:16px"><table class="t"><thead><tr>' +
        '<th>Схема разбиения</th><th>MSE</th><th>σ между обучениями</th><th>ΔMSE к нулю</th>' +
        '<th>Молекул лучше</th><th>RMSE log10</th></tr></thead><tbody>';
      schemes.forEach(function (scheme) {
        var row = crossValidation[scheme] || {};
        var delta = versusZero[scheme] || {};
        var molecules = byMolecule[scheme] || {};
        html += '<tr>' +
          '<td>' + esc(CV_LABELS[scheme] || scheme) + '</td>' +
          '<td>' + fmtInt(row.mse_mean === undefined ? null : Math.round(row.mse_mean)) + '</td>' +
          '<td>' + fmtInt(row.mse_sd === undefined ? null : Math.round(row.mse_sd)) + '</td>' +
          '<td>' + fmtInt(delta.mean_delta_mse === undefined ? null : Math.round(delta.mean_delta_mse)) + '</td>' +
          '<td>' + (molecules.molecules
            ? fmtInt(molecules.molecules_better) + ' из ' + fmtInt(molecules.molecules)
            : '—') + '</td>' +
          '<td>' + fmtNum(row.rmse_log10_mean) + '</td>' +
          '</tr>';
      });
      html += '</tbody></table></div>';
      html += '<p class="note" style="margin-top:12px">' +
        'Случайная схема повторяет то, как разделено само соревнование. Остальные держат в одной части ' +
        'целые каркасы, химические семейства или источники и показывают, насколько модель переносится ' +
        'на незнакомую химию. Отрицательная ΔMSE означает, что подход лучше предсказания нулём ' +
        'на тех же фолдах.</p>';
    }

    html += '<div class="divider"></div>';
    html += '<h2 class="section-title">Область применимости ' +
      '<span class="section-sub">измерено на прогнозах вне обучения</span></h2>';
    html += '<p class="note' + (model.reliability_assessed ? '' : ' note--warn') + '" style="margin-top:16px">' +
      (model.reliability_assessed
        ? 'Надёжность прогноза оценена: на прогнозах вне обучения сходство с обучающими данными действительно отделяет точные прогнозы от неточных. Это всё равно не вероятность того, что прогноз верен.'
        : 'Надёжность прогноза не оценена: показать, что сходство отделяет точные прогнозы от неточных, не удалось. Сходство остаётся мерой похожести на обучающие данные и не является вероятностью того, что прогноз верен.') +
      '</p>';
    html += '<div style="margin-top:16px">' + kvBlock(domain, DOMAIN_LABELS) + '</div>';

    html += '<div class="divider"></div>';
    html += '<h2 class="section-title">Набор данных</h2>';
    html += '<div style="margin-top:16px">' + kvBlock(
      Object.assign({}, dataset.collection || {}, {
        dataset_version: dataset.dataset_version,
        molecules: dataset.n_training_molecules
      }),
      COLLECTION_LABELS,
      ['dataset_version', 'source', 'collected_at', 'molecules', 'vacv_assays', 'publications_scanned',
        'activities_scanned', 'normalised_measurements', 'si_pairs', 'excluded_records']
    ) + '</div>';

    var sources = dataset.sources || [];
    html += '<div class="divider"></div>';
    html += '<h2 class="section-title">Источники <span class="section-sub">публикации, из которых взяты измерения (' +
      fmtInt(sources.length) + ')</span></h2>';
    if (!sources.length) {
      html += '<p class="note" style="margin-top:16px">Сервис не вернул список публикаций.</p>';
    } else {
      html += '<div class="table-wrap" style="margin-top:16px"><table class="t"><thead><tr>' +
        '<th><span class="th-plain">Год</span></th>' +
        '<th><span class="th-plain">Журнал</span></th>' +
        '<th><span class="th-plain">Название</span></th>' +
        '<th><span class="th-plain">DOI</span></th>' +
        '<th><span class="th-plain">PubMed</span></th>' +
        '<th><span class="th-plain">ChEMBL</span></th>' +
        '</tr></thead><tbody>';
      sources.forEach(function (source) {
        html += '<tr>' +
          '<td class="num">' + esc(source.year || '—') + '</td>' +
          '<td>' + esc(source.journal || '—') + '</td>' +
          '<td style="max-width:44ch">' + esc(source.title || '—') + '</td>' +
          '<td>' + doiLinks(source.doi) + '</td>' +
          '<td class="num">' + (source.pubmed_id
            ? '<a href="https://pubmed.ncbi.nlm.nih.gov/' + encodeURIComponent(String(source.pubmed_id)) + '/" target="_blank" rel="noopener noreferrer">' + esc(source.pubmed_id) + '</a>'
            : '—') + '</td>' +
          '<td class="mono">' + esc(source.document_chembl_id || '—') + '</td>' +
          '</tr>';
      });
      html += '</tbody></table></div>';
    }

    var exclusions = dataset.exclusion_summary || {};
    var exclusionKeys = Object.keys(exclusions);
    html += '<div class="divider"></div>';
    html += '<h2 class="section-title">Что исключено <span class="section-sub">и почему</span></h2>';
    if (!exclusionKeys.length) {
      html += '<p class="note" style="margin-top:16px">Сервис не вернул список исключений.</p>';
    } else {
      html += '<div class="table-wrap" style="margin-top:16px"><table class="t"><thead><tr>' +
        '<th><span class="th-plain">Причина</span></th>' +
        '<th><span class="th-plain">Записей</span></th>' +
        '</tr></thead><tbody>';
      exclusionKeys.forEach(function (reason) {
        html += '<tr><td style="max-width:60ch">' + esc(reason) + '</td>' +
          '<td class="num">' + fmtInt(exclusions[reason]) + '</td></tr>';
      });
      html += '</tbody></table></div>';
    }

    var limitations = dataset.limitations || model.limitations || [];
    html += '<div class="divider"></div>';
    html += '<h2 class="section-title">Ограничения <span class="section-sub">текст из описания модели, как его отдаёт сервис</span></h2>';
    html += '<ul style="margin-top:16px;display:flex;flex-direction:column;gap:12px">';
    limitations.forEach(function (limitation) {
      html += '<li class="note">' + esc(limitation) + '</li>';
    });
    html += '</ul>';

    setPanel(panel, 'ok', html);
  }

  // ───────────────────────── события ─────────────────────────

  function bind() {
    // меню и экраны
    Array.prototype.forEach.call(document.querySelectorAll('.nav__item'), function (item) {
      item.addEventListener('click', function () { showScreen(item.getAttribute('data-screen')); });
    });

    $('sidebarToggle').addEventListener('click', function () {
      setCollapsed(document.documentElement.getAttribute('data-sidebar') !== 'collapsed');
    });
    $('menuBtn').addEventListener('click', function () {
      if ($('sidebar').classList.contains('is-open')) closeDrawer(); else openDrawer();
    });
    $('scrim').addEventListener('click', closeDrawer);
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && $('sidebar').classList.contains('is-open')) closeDrawer();
    });

    // тема
    Array.prototype.forEach.call(document.querySelectorAll('[data-theme-value]'), function (button) {
      button.addEventListener('click', function () { setTheme(button.getAttribute('data-theme-value')); });
    });

    // анализ
    $('runSingle').addEventListener('click', runSingle);
    $('clearSingle').addEventListener('click', function () {
      $('smilesInput').value = '';
      setFieldError('smilesInput', 'smilesError', '');
      state.single.result = null;
      state.single.error = null;
      renderSingle();
      $('smilesInput').focus();
    });
    $('smilesInput').addEventListener('keydown', function (event) {
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { event.preventDefault(); runSingle(); }
    });
    $('smilesInput').addEventListener('input', function () {
      if ($('smilesError').textContent) setFieldError('smilesInput', 'smilesError', '');
    });

    // Делегируем: такие кнопки есть и в разметке, и в пустых состояниях, которые перерисовываются.
    document.addEventListener('click', function (event) {
      if (!event.target.closest) return;
      var example = event.target.closest('[data-example]');
      if (example) {
        $('smilesInput').value = example.getAttribute('data-example');
        setFieldError('smilesInput', 'smilesError', '');
        showScreen('analysis');
        runSingle();
        return;
      }
      var link = event.target.closest('[data-screen-link]');
      if (link) showScreen(link.getAttribute('data-screen-link'));
    });

    Array.prototype.forEach.call(document.querySelectorAll('[data-input-mode]'), function (button) {
      button.addEventListener('click', function () {
        var mode = button.getAttribute('data-input-mode');
        Array.prototype.forEach.call(document.querySelectorAll('[data-input-mode]'), function (other) {
          other.classList.toggle('is-active', other === button);
        });
        if (mode === 'csv') {
          showScreen('batch');
          // сегмент возвращаем в исходное положение: экран ввода CSV — отдельный раздел
          Array.prototype.forEach.call(document.querySelectorAll('[data-input-mode]'), function (other) {
            other.classList.toggle('is-active', other.getAttribute('data-input-mode') === 'single');
          });
        }
      });
    });

    // пакетный расчёт
    $('runBatch').addEventListener('click', function () {
      runBatch(parsePastedSmiles($('batchText').value), 'Список введён вручную');
    });
    $('clearBatch').addEventListener('click', function () {
      $('batchText').value = '';
      $('fileInput').value = '';
      $('batchSource').textContent = '';
      setFieldError('batchText', 'batchError', '');
      state.batch.rows = [];
      state.batch.selected = {};
      state.batch.error = null;
      state.batch.errorTitle = null;
      state.batch.sort = { key: null, dir: 1 };
      renderBatch();
    });

    var dropzone = $('dropzone');
    dropzone.addEventListener('click', function () { $('fileInput').click(); });
    dropzone.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('fileInput').click(); }
    });
    ['dragenter', 'dragover'].forEach(function (name) {
      dropzone.addEventListener(name, function (event) {
        event.preventDefault();
        dropzone.classList.add('is-over');
      });
    });
    ['dragleave', 'dragend', 'drop'].forEach(function (name) {
      dropzone.addEventListener(name, function () { dropzone.classList.remove('is-over'); });
    });
    dropzone.addEventListener('drop', function (event) {
      event.preventDefault();
      var file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) uploadCsv(file);
    });
    $('fileInput').addEventListener('change', function (event) {
      var file = event.target.files && event.target.files[0];
      if (file) uploadCsv(file);
    });
    // перетаскивание мимо зоны не должно открывать файл поверх приложения
    window.addEventListener('dragover', function (event) { event.preventDefault(); });
    window.addEventListener('drop', function (event) { event.preventDefault(); });

    // действия внутри панелей
    document.addEventListener('click', function (event) {
      var target = event.target.closest ? event.target.closest('[data-action],[data-sort],[data-copy],[data-remove]') : null;
      if (!target) return;

      var copy = target.getAttribute('data-copy');
      if (copy !== null) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(copy).then(function () { toast('Скопировано'); },
            function () { toast('Браузер не дал скопировать'); });
        } else {
          toast('Браузер не поддерживает копирование');
        }
        return;
      }

      var sortKey = target.getAttribute('data-sort');
      if (sortKey) {
        if (state.batch.sort.key === sortKey) state.batch.sort.dir = state.batch.sort.dir === 1 ? -1 : 1;
        else state.batch.sort = { key: sortKey, dir: 1 };
        renderBatch();
        return;
      }

      var removeKey = target.getAttribute('data-remove');
      if (removeKey) {
        state.worklist = state.worklist.filter(function (item) { return item.key !== removeKey; });
        saveWorklist();
        renderWorklistCount();
        renderWorklist();
        toast('Убрано из подборки');
        return;
      }

      switch (target.getAttribute('data-action')) {
        case 'recheck':
          checkHealth();
          break;
        case 'reload-data':
          state.data.loaded = false;
          loadData();
          break;
        case 'export-batch':
          exportBatch();
          break;
        case 'export-worklist':
          exportWorklist();
          break;
        case 'compare': {
          var chosen = selectedRows();
          if (chosen.length) openCompare(chosen);
          break;
        }
        case 'batch-to-worklist':
          addToWorklist(selectedRows(), state.batch.modelVersion, state.batch.datasetVersion);
          break;
        case 'single-to-worklist':
          if (state.single.result) {
            addToWorklist([state.single.result], state.single.modelVersion, state.single.datasetVersion);
          }
          break;
        case 'worklist-clear':
          if (state.worklist.length && window.confirm('Убрать все молекулы из подборки?')) {
            state.worklist = [];
            saveWorklist();
            renderWorklistCount();
            renderWorklist();
          }
          break;
        default:
          break;
      }
    });

    // выбор строк таблицы
    $('batchPanel').addEventListener('change', function (event) {
      var input = event.target;
      if (input.getAttribute('data-action') === 'select-all') {
        if (input.checked) {
          state.batch.rows.forEach(function (row) { state.batch.selected[row.__key] = true; });
        } else {
          state.batch.selected = {};
        }
        renderBatch();
        return;
      }
      var rowKey = input.getAttribute('data-row');
      if (rowKey) {
        if (input.checked) state.batch.selected[rowKey] = true;
        else delete state.batch.selected[rowKey];
        renderBatch();
      }
    });

    // заметки подборки
    var saveNote = debounce(function (key, value) {
      var item = state.worklist.filter(function (entry) { return entry.key === key; })[0];
      if (!item) return;
      item.note = value;
      saveWorklist();
    }, 400);
    $('worklistPanel').addEventListener('input', function (event) {
      var key = event.target.getAttribute && event.target.getAttribute('data-note');
      if (key) saveNote(key, event.target.value);
    });

    // сравнение
    $('compareClose').addEventListener('click', closeCompare);
    $('compareToWorklist').addEventListener('click', function () {
      addToWorklist(compareRows, state.batch.modelVersion, state.batch.datasetVersion);
      closeCompare();
    });

    // адрес со ссылкой на раздел: «назад», «вперёд» и ручная правка адреса
    window.addEventListener('hashchange', function () {
      var name = (location.hash || '').replace('#', '');
      if (SCREEN_TITLES[name] && name !== state.screen) showScreen(name);
    });

    // состояние сервиса обновляем при возвращении на вкладку
    window.addEventListener('focus', function () { checkHealth(); });

    window.addEventListener('resize', debounce(function () {
      if (!isNarrow()) closeDrawer();
    }, 150));
  }

  // ───────────────────────── запуск ─────────────────────────

  function start() {
    markThemeButtons();
    setCollapsed(document.documentElement.getAttribute('data-sidebar') === 'collapsed');
    loadWorklist();
    renderWorklistCount();
    bind();

    var hash = (location.hash || '').replace('#', '');
    showScreen(SCREEN_TITLES[hash] ? hash : 'analysis');

    renderSingle();
    renderBatch();
    renderWorklist();
    renderData();
    checkHealth();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();

})();
