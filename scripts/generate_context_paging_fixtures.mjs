#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const EXPECTED_BASELINE = "4409e5e4ffa0b626c11e72afe666f0b6e55a0e5f";
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT = join(ROOT, "tests/fixtures/context_paging");
const DATE = "2026-10-19T00:00:00.000Z";
const USAGE = {
  input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function userMessage(content, timestamp = 0) {
  return { role: "user", content, timestamp };
}

function assistantMessage(content, stopReason = "stop", timestamp = 1) {
  return {
    role: "assistant", api: "openai-completions", provider: "fixture", model: "fixture",
    content, usage: USAGE, stopReason, timestamp,
  };
}

function resultMessage(toolCallId, text, isError = false, toolName = "read") {
  return {
    role: "toolResult", toolCallId, toolName, content: [{ type: "text", text }], isError, timestamp: 2,
  };
}

function entry(id, message) {
  return { type: "message", id, parentId: null, timestamp: DATE, message };
}

function userEntry(id, content) {
  return entry(id, userMessage(content));
}

function assistantEntry(id, content, stopReason = "stop") {
  return entry(id, assistantMessage(content, stopReason));
}

function resultEntry(id, toolCallId, text, isError = false, toolName = "read") {
  return entry(id, resultMessage(toolCallId, text, isError, toolName));
}

function ordinary(id, text = "answer") {
  return assistantEntry(id, [{ type: "text", text }]);
}

function call(id, name, arguments_ = {}) {
  return { type: "toolCall", id, name, arguments: arguments_ };
}

function userItem(id, sequence, content) {
  return { id, kind: "user", sequence, timestamp: DATE, userMessage: userMessage(content, sequence) };
}

function turnItem(id, sequence, content, results = [], metadata = {}) {
  return {
    id, kind: "modelTurn", sequence, timestamp: DATE,
    assistantMessage: assistantMessage(content, "toolUse", sequence),
    toolResults: results,
    metadata: { tools: [], files: [], failed: false, ...metadata },
  };
}

function capture(action) {
  try {
    const value = action();
    return value === undefined ? { undefined: true } : { value };
  } catch (error) {
    return { error: errorRecord(error) };
  }
}

async function captureAsync(action) {
  try {
    const value = await action();
    return value === undefined ? { undefined: true } : { value };
  } catch (error) {
    return { error: errorRecord(error) };
  }
}

function errorRecord(error) {
  return {
    name: error instanceof Error ? error.name : "Error",
    ...(error && typeof error === "object" && "code" in error ? { code: error.code } : {}),
    message: error instanceof Error ? error.message : String(error),
  };
}

function byId(left, right) {
  return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
}

function settingsFixtures(api) {
  const paging = (tokenBudget, trimToTokens, enabled) => ({
    contextPaging: {
      ...(enabled === undefined ? {} : { enabled }),
      ...(tokenBudget === undefined ? {} : { tokenBudget }),
      ...(trimToTokens === undefined ? {} : { trimToTokens }),
    },
  });
  const cases = [
    ["defaults", { globalSettings: {}, projectTrusted: false }],
    ["trusted-project-precedence", {
      globalSettings: paging(128000, 80000, false),
      projectSettings: paging(64000, 40000, true), projectTrusted: true,
    }],
    ["untrusted-project-ignored", {
      globalSettings: paging(64000, 30000, false),
      projectSettings: paging(128000, 90000, true), projectTrusted: false,
    }],
    ["global-target-survives-lower-project-budget", {
      globalSettings: paging(128000, 80000),
      projectSettings: paging(64000), projectTrusted: true,
    }],
    ["invalid-project-falls-through", {
      globalSettings: paging(64000, 30000),
      projectSettings: paging(32000, 2 ** 53), projectTrusted: true,
    }],
    ["invalid-global-uses-adaptive-target", {
      globalSettings: paging(64000, 0), projectTrusted: false,
    }],
    ["maximum-safe-integer-accepted", {
      globalSettings: paging(2 ** 53 - 1, 2 ** 53 - 1), projectTrusted: false,
    }],
    ["above-safe-integer-rejected", {
      globalSettings: paging(2 ** 53, 2 ** 53), projectTrusted: false,
    }],
    ["non-integer-rejected", {
      globalSettings: paging(64000.5, 1.5), projectTrusted: false,
    }],
  ];
  const fixtures = cases.map(([name, sources]) => ({
    id: `settings-${name}`,
    input: sources,
    expected: capture(() => api.resolveContextPagingSettings(sources)),
  }));
  for (const budget of [1, 2 ** 53 - 4, 2 ** 53 - 1]) {
    fixtures.push({
      id: `settings-adaptive-${budget}`,
      input: { tokenBudget: budget },
      expected: capture(() => api.defaultTrimToTokens(budget)),
    });
  }
  return { schemaVersion: 1, fixtures: fixtures.sort(byId) };
}

function historyFixtures(api) {
  const fixtures = [];
  const add = (id, entries, options = {}) => {
    const projection = capture(() => api.projectActiveBranch(entries, options));
    const expected = { projection };
    if (projection.value) {
      expected.public = projection.value.map((item) => api.recoveryHistoryItem(item) ?? null);
    }
    fixtures.push({ id: `history-${id}`, input: { entries, options }, expected });
  };
  const work = (id, callId = `call-${id}`, name = "read") => assistantEntry(id, [
    call(callId, name, { path: `src/${id}.ts` }),
  ], "toolUse");

  add("complete-exchange", [
    userEntry("user-1", "inspect source"),
    assistantEntry("turn-1", [call("call-1", "read", { path: "src/a.ts" }), call("call-2", "read", { path: "src/b.ts" })], "toolUse"),
    resultEntry("result-2", "call-2", "second", true),
    resultEntry("result-1", "call-1", "first"),
  ]);
  add("newest-incomplete-tail-omitted", [
    userEntry("user-1", "first"), ordinary("turn-1"), userEntry("user-2", "second"), work("turn-2"),
  ]);
  add("older-incomplete-rejected", [work("turn-1"), userEntry("user-2", "continue")]);
  add("orphan-result-rejected", [resultEntry("result-1", "missing", "orphan")]);
  add("duplicate-result-rejected", [
    work("turn-1", "call-1"), resultEntry("result-1", "call-1", "first"), resultEntry("result-2", "call-1", "second"),
  ]);
  add("mismatched-result-rejected", [work("turn-1", "call-1"), resultEntry("result-1", "other", "wrong")]);
  for (const stopReason of ["aborted", "error"]) {
    add(`${stopReason}-retained-with-failure`, [
      assistantEntry(`turn-${stopReason}`, [call("call-1", "read"), call("call-2", "read")], stopReason),
      resultEntry("result-1", "call-1", "partial"), userEntry("user-next", "continue"),
    ]);
  }
  add("restore-incomplete-retained", [
    assistantEntry("turn-restored", [call("call-1", "read"), call("call-2", "read")], "toolUse"),
    resultEntry("result-1", "call-1", "partial"), userEntry("user-next", "continue"),
  ], { recoverIncomplete: true });
  add("denied-and-retry-failures", [
    assistantEntry("turn-failures", [call("denied", "bash"), call("retry", "read")], "toolUse"),
    resultEntry("result-denied", "denied", "Tool execution denied", true, "bash"),
    resultEntry("result-retry", "retry", "Retry prompt: correct the input", true),
  ]);
  add("length-capped-is-not-failure", [
    assistantEntry("turn-length", [{ type: "text", text: "truncated" }], "length"),
  ]);

  add("public-privacy-and-original-indices", [
    assistantEntry("turn-privacy", [
      { type: "thinking", thinking: "private-reasoning", thinkingSignature: "private-signature" },
      { type: "text", text: "Public answer", textSignature: "private-text-signature" },
      { ...call("ordinary", "read", { path: "src/public.ts", thinking: "application-owned" }), thoughtSignature: "private-tool-signature" },
      call("paging", "load_history", { hidden: "paging-argument" }),
      { type: "futurePrivateBlock", payload: "private-unknown" },
    ], "toolUse"),
    resultEntry("result-ordinary", "ordinary", "Public result"),
    resultEntry("result-paging", "paging", "Hidden paging result", false, "load_history"),
  ]);
  add("paging-only-turn-excluded", [
    work("turn-paging", "paging-call", "search_history"),
    resultEntry("result-paging", "paging-call", "hidden", false, "search_history"),
  ]);

  const wrapperEntries = (nestedCalls, mixed) => {
    const wrapped = {
      ...resultMessage("wrapper-call", "hidden-wrapper", true, "codemode"),
      ...(nestedCalls === undefined ? {} : { nestedCalls }),
      details: { nestedCalls: { complete: true, calls: [{ id: "n1", name: "load_history", status: "ok" }] } },
    };
    return [
      assistantEntry("turn-wrapper", [
        { type: "thinking", thinking: "private" },
        call("wrapper-call", "codemode", { code: "hiddenScript()" }),
        { type: "text", text: "Public decision" },
        ...(mixed ? [call("ordinary-call", "read", { path: "src/app.ts" })] : []),
      ], "toolUse"),
      entry("result-wrapper", wrapped),
      ...(mixed ? [resultEntry("result-ordinary", "ordinary-call", "ordinary-evidence")] : []),
    ];
  };
  const onlyPaging = { complete: true, calls: [{ id: "n1", name: "load_history", status: "ok" }] };
  add("nested-paging-wrapper-mixed-turn", wrapperEntries(onlyPaging, true));
  add("nested-wrapper-malformed-retained", wrapperEntries({ complete: false, calls: onlyPaging.calls }, false));
  add("nested-wrapper-absent-retained", wrapperEntries(undefined, false));
  return { schemaVersion: 1, fixtures: fixtures.sort(byId) };
}

function selectionFixtures(api, fixtureApi) {
  const workEntries = (id, output = `${id} output`) => [
    assistantEntry(`turn-${id}`, [call(`call-${id}`, "read", { path: `${id}.txt` })], "toolUse"),
    resultEntry(`result-${id}`, `call-${id}`, output),
  ];
  const completedTurn = (id) => [userEntry(`user-${id}`, `${id} request`), ordinary(`turn-${id}`, `${id} answer`)];
  const entriesFor = (kind) => {
    if (kind === "completed") return [
      ...completedTurn("old-A"), ...completedTurn("old-B"), ...completedTurn("old-C"),
      userEntry("user-live", "live request"), ...workEntries("live-1"), ...workEntries("live-2"),
    ];
    if (kind === "active") return [
      userEntry("user-live", "live request"),
      ...workEntries("live-1"), ...workEntries("live-2"), ...workEntries("live-3"), ...workEntries("live-4"),
    ];
    if (kind === "keyless") return [
      ...completedTurn("old-A"),
      { type: "custom_message", id: "custom-old", parentId: null, timestamp: DATE,
        customType: "fixture", display: true, details: {}, content: "keyless old request" },
      userEntry("user-live", "live request"),
    ];
    return [userEntry("user-live", "live request"), ...workEntries("old"), ...workEntries("newest")];
  };
  const makeInput = (kind, options = {}) => {
    const fixture = fixtureApi.fixtureFromEntries(entriesFor(kind));
    const messageTokens = options.messageTokens ?? fixture.input.messages.map(() => 200);
    const residentTokens = 0;
    return {
      ...fixture.input,
      tokenBudget: options.tokenBudget ?? 1000,
      trimToTokens: options.trimToTokens ?? 600,
      modelContextWindow: options.modelContextWindow ?? options.tokenBudget ?? 1000,
      contextTokens: options.contextTokens ?? residentTokens + messageTokens.reduce((sum, value) => sum + value, 0),
      tokenEstimates: { residentTokens, messageTokens },
      ...(options.rawHistoryItems === null ? { rawHistoryItems: undefined } : {}),
    };
  };
  const fixtures = [];
  const add = (id, input, replay) => {
    const expected = capture(() => api.selectContext(input));
    const fixture = { id: `selection-${id}`, input, expected };
    if (replay && expected.value?.cutState) {
      const replayInput = { ...input, ...replay, cutState: expected.value.cutState };
      fixture.replay = { input: replayInput, expected: capture(() => api.selectContext(replayInput)) };
    }
    fixtures.push(fixture);
  };
  add("stable-completed-turn", makeInput("completed", {
    tokenBudget: 1600, trimToTokens: 1400, modelContextWindow: 1600,
  }), { contextTokens: 1000 });
  add("partial-active-turn", makeInput("active"), {});
  add("keyless-backward-snap", makeInput("keyless", {
    tokenBudget: 700, trimToTokens: 100, modelContextWindow: 700,
  }), {});
  add("fallback-without-raw-history", makeInput("completed", { rawHistoryItems: null }));
  add("model-capped-target", makeInput("completed", { tokenBudget: 1200, trimToTokens: 750, modelContextWindow: 800 }));
  const protectedCosts = [20, 20, 20, 20, 180];
  add("protected-overflow", makeInput("protected", {
    messageTokens: protectedCosts, tokenBudget: 100, trimToTokens: 50, modelContextWindow: 400,
  }));
  add("protected-recovery", makeInput("protected", {
    messageTokens: protectedCosts, tokenBudget: 100, trimToTokens: 50, modelContextWindow: 200,
  }));
  return { schemaVersion: 1, fixtures: fixtures.sort(byId) };
}

function noticeFixtures(api) {
  const makeTurns = (specs) => specs.flatMap((spec, index) => [
    userEntry(`u${index + 1}`, spec.request ?? `U${index + 1} request`),
    ordinary(`r${index + 1}`, spec.reply ?? `R${index + 1} final reply`),
  ]);
  const fixtures = [];
  const add = (id, entries, budgetTokens, maxCharacters) => {
    const items = api.projectActiveBranch(entries);
    const lookup = new api.SelectionHistoryLookup(items);
    const conversation = new api.EvictedConversation(lookup);
    for (let index = 0; index < entries.length; index += 2) {
      conversation.addTurn([entries[index].message, entries[index + 1].message]);
    }
    const input = {
      entries, configuredTokenBudget: 128000, effectiveBudgetTokens: budgetTokens,
      ...(maxCharacters === undefined ? {} : { maxCharacters }),
    };
    fixtures.push({
      id: `notice-${id}`, input,
      expected: capture(() => api.pagingNotice(budgetTokens, entries.at(-1)?.id, undefined, conversation, maxCharacters)),
    });
  };
  add("newest-three-replies", makeTurns(Array.from({ length: 6 }, () => ({}))), 128000);
  add("utf16-clipping-and-handles", makeTurns([
    { request: `HEAD ${"y".repeat(5994)}😀 TAIL_MARKER`, reply: `REPLY ${"z".repeat(10000)}` },
    {}, {},
  ]), 128000);
  add("low-effective-budget-cap", makeTurns(Array.from({ length: 14 }, (_, index) => ({
    request: `U${index + 1} request ${"q".repeat(5000)}`,
  }))), 20000);
  add("minimum-useful-clip", makeTurns([{ request: "x".repeat(5000) }, {}, {}]), 20000, 1000);
  return { schemaVersion: 1, fixtures: fixtures.sort(byId) };
}

function navigatorFixtures(api) {
  const items = [
    userItem("user-1", 0, "Please find the needle"),
    turnItem("turn-content", 1, [{ type: "text", text: "needle needle" }]),
    turnItem("turn-read", 2, [
      { type: "text", text: "other" },
      call("call-read", "read", { path: "src/nested/needle.ts" }),
    ], [resultMessage("call-read", "needle result")], { tools: ["read"], files: ["src/nested/needle.ts"], failed: true }),
    turnItem("turn-paging", 3, [call("paging", "read_context_output")], [], { tools: ["read_context_output"] }),
    userItem("user-unicode", 4, "请分页显示历史记录"),
  ];
  const fixtures = [];
  const add = (id, action) => {
    const navigator = new api.HistoryNavigator(items);
    fixtures.push({
      id: `navigator-${id}`, input: { items, action },
      expected: capture(() => {
        if (action.type === "search") return navigator.search(action.input);
        if (action.type === "browse") return navigator.browse(action.input);
        return navigator.load(action.input);
      }),
    });
  };
  add("search-ranking", { type: "search", input: { query: "NEEDLE" } });
  add("search-filters", { type: "search", input: {
    query: "needle", files: ["src/nested/needle.ts"], tools: ["read"], failed: true, limit: 5,
  } });
  add("search-unicode", { type: "search", input: { query: "分页" } });
  add("browse-unanchored-backward", { type: "browse", input: { direction: "backward", count: 3 } });
  add("browse-forward-sequence", { type: "browse", input: { sequence: 1, direction: "forward", count: 2 } });
  add("browse-around-stride", { type: "browse", input: { historyId: "turn-read", direction: "around", count: 3, stride: 2 } });
  add("load-requested-order", { type: "load", input: ["turn-read", "user-1"] });
  add("paging-turn-excluded", { type: "browse", input: { direction: "forward", count: 10 } });
  add("unknown-load-is-atomic", { type: "load", input: ["user-1", "missing"] });
  return { schemaVersion: 1, fixtures: fixtures.sort(byId) };
}

async function recoveryFixtures(api) {
  const fixtures = [];
  const registered = (items) => {
    const tools = [];
    const navigator = new api.HistoryNavigator(items);
    api.registerContextPagingTools({ registerTool(tool) { tools.push(tool); } }, {
      isEnabled: () => true,
      snapshot: () => ({ allItems: items, navigator }),
    });
    return async (name, params) => {
      const tool = tools.find((candidate) => candidate.name === name);
      if (!tool) throw new Error(`Unknown fixture tool ${name}`);
      return tool.execute("fixture-call", params, undefined, undefined, {});
    };
  };
  const add = async (id, items, tool, params, transform) => {
    const execute = registered(items);
    const expected = await captureAsync(() => execute(tool, params));
    fixtures.push({ id: `recovery-${id}`, input: { items, tool, params }, expected: transform ? transform(expected) : expected });
  };
  await add("load-smallest-first-and-defer", [
    userItem("a", 0, "a".repeat(7000)), userItem("b", 1, "b".repeat(3000)), userItem("c", 2, "c".repeat(7000)),
  ], "load_history", { historyIds: ["a", "b", "c"] });
  await add("byte-budget-and-continuation", [
    userItem("large", 0, "0123456789".repeat(4000)),
  ], "read_context_output", { historyId: "large", part: "/userMessage/content" });
  await add("deferred-part-descriptions", [
    userItem("large", 0, "large ".repeat(8000)),
  ], "read_context_output", { historyId: "large", mode: "parts", limit: 2 });
  await add("utf16-page-boundary", [
    userItem("unicode", 0, `A${"😀".repeat(200)}B漢字`),
  ], "read_context_output", { historyId: "unicode", part: "/userMessage/content", limit: 101 });

  const selectorItems = [turnItem("turn", 0, [
    { type: "thinking", thinking: "private" },
    { type: "text", text: "Public answer" },
    call("call", "bash", { command: "echo hi" }),
  ], [resultMessage("call", "output", false, "bash")], { tools: ["bash"] })];
  await add("assistant-selector-default", selectorItems, "read_context_output", { historyId: "turn", source: "assistant" });
  await add("tool-result-selector-default", selectorItems, "read_context_output", { historyId: "turn", source: "toolResult" });
  await add("exact-json-part", selectorItems, "read_context_output", { historyId: "turn", part: "/assistantMessage/content/2" });
  await add("private-part-error", selectorItems, "read_context_output", { historyId: "turn", part: "/assistantMessage/content/0" });

  const image = { type: "image", data: "SU1BR0VEQVRB", mimeType: "image/png" };
  const imageItems = [userItem("image", 0, [{ type: "text", text: "Inspect this" }, image])];
  await add("small-image-load-always-defers", imageItems, "load_history", { historyIds: ["image"] });
  await add("image-text-is-metadata-only", imageItems, "read_context_output", {
    historyId: "image", part: "/userMessage/content/1", mode: "text",
  });
  await add("image-mode-m5-error", imageItems, "read_context_output", {
    historyId: "image", part: "/userMessage/content/1", mode: "image",
  }, (observed) => {
    if (!observed.value?.content?.some((part) => part.type === "image")) {
      throw new Error("The pinned baseline did not accept the valid image handle");
    }
    return {
      value: {
        baselineValidation: "valid native image handle",
        ds4Error: { name: "EngineError", message: "image recovery requires M5 vision support" },
      },
    };
  });
  return { schemaVersion: 1, fixtures: fixtures.sort(byId) };
}

function manifest(groups) {
  const coverage = [
    { contract: "settings precedence and safe-integer boundaries", fixtures: groups.settings.fixtures.map((item) => item.id),
      sourceTests: ["tests/settings.test.ts"] },
    { contract: "exchange projection, privacy, exclusions, interruptions, and failure shapes", fixtures: groups.history.fixtures.map((item) => item.id),
      sourceTests: ["tests/history.test.ts", "tests/public-history.test.ts", "tests/nested-paging-exclusion.test.ts"] },
    { contract: "stable, partial, keyless, fallback, model-capped, and protected additive-cost selection", fixtures: groups.selection.fixtures.map((item) => item.id),
      sourceTests: ["tests/context-cut.test.ts", "tests/context-policy.test.ts", "tests/fixtures/context-cut.ts"] },
    { contract: "notice conversation selection, clipping, handles, and effective-budget caps", fixtures: groups.notice.fixtures.map((item) => item.id),
      sourceTests: ["tests/conversation-notice.test.ts"] },
    { contract: "search, browse, load ordering, filters, and exclusions", fixtures: groups.navigator.fixtures.map((item) => item.id),
      sourceTests: ["tests/navigator.test.ts", "tests/navigator-previews.test.ts"] },
    { contract: "reply byte limits, deferred parts, UTF-16 selectors, image metadata/deferral, and the M5 error", fixtures: groups.recovery.fixtures.map((item) => item.id),
      sourceTests: ["tests/bounded-recovery.test.ts", "tests/read-back.test.ts", "tests/recovery-parts.test.ts", "tests/output-pages.test.ts"] },
  ];
  return {
    schemaVersion: 1,
    baselineCommit: EXPECTED_BASELINE,
    generatorCommand: "node scripts/generate_context_paging_fixtures.mjs /tmp/pi-context-paging-4409e5e",
    files: Object.keys(groups).sort().map((name) => `${name}.json`),
    coverage,
  };
}

async function main() {
  const argument = process.argv[2];
  if (!argument) {
    console.error("Usage: node scripts/generate_context_paging_fixtures.mjs <baseline-checkout>");
    process.exitCode = 2;
    return;
  }
  const baseline = resolve(argument);
  const actual = execFileSync("git", ["-C", baseline, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  if (actual !== EXPECTED_BASELINE) {
    console.error(`Baseline commit mismatch: expected ${EXPECTED_BASELINE}, actual ${actual}`);
    process.exitCode = 1;
    return;
  }
  const importBaseline = (path) => import(pathToFileURL(join(baseline, path)).href);
  const [settings, history, recoveryContent, contextPolicy, fixtureApi, selectionHistory, notice, navigator, tools] = await Promise.all([
    importBaseline("src/settings.ts"), importBaseline("src/history.ts"), importBaseline("src/recovery-content.ts"),
    importBaseline("src/context-policy.ts"), importBaseline("tests/fixtures/context-cut.ts"),
    importBaseline("src/selection-history.ts"), importBaseline("src/paging-notice.ts"),
    importBaseline("src/navigator.ts"), importBaseline("src/tools.ts"),
  ]);
  const groups = {
    settings: settingsFixtures(settings),
    history: historyFixtures({ ...history, ...recoveryContent }),
    selection: selectionFixtures(contextPolicy, fixtureApi),
    notice: noticeFixtures({ ...history, ...selectionHistory, ...notice }),
    navigator: navigatorFixtures(navigator),
    recovery: await recoveryFixtures({ ...navigator, ...tools }),
  };
  mkdirSync(OUTPUT, { recursive: true });
  for (const [name, group] of Object.entries(groups)) {
    writeFileSync(join(OUTPUT, `${name}.json`), `${JSON.stringify(group, null, 2)}\n`);
  }
  writeFileSync(join(OUTPUT, "manifest.json"), `${JSON.stringify(manifest(groups), null, 2)}\n`);
  const count = Object.values(groups).reduce((sum, group) => sum + group.fixtures.length, 0);
  console.log(`Generated ${count} fixtures from ${actual}`);
}

await main();
