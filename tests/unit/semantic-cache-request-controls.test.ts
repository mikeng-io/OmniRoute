import test, { mock } from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import {
  generateSignature,
  outputContractOf,
  isCacheableForWrite,
} from "../../src/lib/semanticCache.ts";
import {
  generateDirectHash,
  getSemanticCacheManager,
} from "../../open-sse/services/cache/semanticCacheManager.ts";
import { storeSemanticCacheResponse } from "../../open-sse/handlers/chatCore/semanticCacheStore.ts";
import { storeStreamingSemanticCacheResponse } from "../../open-sse/handlers/chatCore/streamingSemanticCacheStore.ts";

const messages = [{ role: "user", content: "Return two plus two." }];
const signature = (body: Record<string, unknown>, key = "synthetic-key") =>
  generateSignature("synthetic-model", messages, 0, 1, key, outputContractOf(body));
const direct = (body: Record<string, unknown>) =>
  generateDirectHash("synthetic-model", messages, 0, 1, {}, outputContractOf(body));

test("versioned signatures cannot read old incomplete plain-request entries", () => {
  const old = crypto
    .createHash("sha256")
    .update(
      JSON.stringify({
        model: "synthetic-model",
        messages,
        temperature: 0,
        top_p: 1,
      })
    )
    .digest("hex");
  assert.notEqual(signature({}), `synthetic-key.${old}`);
});

test("effort, each token cap, and supported response controls separate identities", () => {
  const controls = [
    { reasoning_effort: "low" },
    { reasoning: { effort: "low" } },
    { max_tokens: 4096 },
    { max_completion_tokens: 4096 },
    { max_output_tokens: 4096 },
    { effort: "low" },
    { thinking: false },
    { thinking: { type: "enabled", budget_tokens: 1024 } },
    { parallel_tool_calls: false },
    { functions: [{ name: "sum" }] },
    { function_call: "none" },
    { n: 2 },
    { logprobs: true },
    { top_logprobs: 2 },
    { text: { verbosity: "low" } },
    { verbosity: "low" },
    { output_config: { effort: "low" } },
    { generationConfig: { maxOutputTokens: 2048 } },
    { stop_sequences: ["END"] },
    { instructions: "Be brief." },
    { seed: 0 },
    { stop: "" },
    { presence_penalty: 0 },
    { frequency_penalty: 0 },
    { repetition_penalty: 1.2 },
    { logit_bias: { "10": -1 } },
    { top_k: 0 },
    { response_format: { type: "json_object" } },
    { tool_choice: "none" },
    { tools: [{ type: "web_search" }] },
  ];
  for (const control of controls) {
    assert.notEqual(signature(control), signature({}), JSON.stringify(control));
    assert.notEqual(direct(control), direct({}), JSON.stringify(control));
  }
  for (const cap of ["max_tokens", "max_completion_tokens", "max_output_tokens"]) {
    assert.notEqual(signature({ [cap]: 4096 }), signature({ [cap]: 2048 }));
  }
});

test("nested key order is irrelevant while tool policy and array order remain significant", () => {
  const a = {
    reasoning: { effort: "low", summary: "auto" },
    tools: [
      {
        type: "function",
        function: {
          name: "sum",
          strict: true,
          parameters: { type: "object", properties: { a: { type: "number" } } },
        },
      },
    ],
  };
  const b = {
    tools: [
      {
        function: {
          parameters: { properties: { a: { type: "number" } }, type: "object" },
          strict: true,
          name: "sum",
        },
        type: "function",
      },
    ],
    reasoning: { summary: "auto", effort: "low" },
  };
  assert.equal(signature(a), signature(b));
  assert.equal(direct(a), direct(b));
  assert.notEqual(
    signature(a),
    signature({
      ...a,
      tools: [{ ...a.tools[0], function: { ...a.tools[0].function, strict: false } }],
    })
  );
  assert.notEqual(
    signature({ tools: [{ type: "web_search", search_context_size: "low" }] }),
    signature({ tools: [{ type: "web_search", search_context_size: "high" }] })
  );
  assert.notEqual(signature({ stop: ["A", "B"] }), signature({ stop: ["B", "A"] }));
});

test("identical/default requests hit, API namespaces stay separate, transport metadata is ignored", () => {
  assert.equal(signature({}), signature({ reasoning_effort: undefined }));
  assert.equal(
    signature({}),
    signature({ stream: true, request_id: "synthetic", authorization: "unused" })
  );
  assert.equal(signature({ max_tokens: 4096 }), signature({ max_tokens: 4096 }));
  assert.notEqual(signature({}, "synthetic-a"), signature({}, "synthetic-b"));
});

test("both stores use the read identity; complete replies cannot cross controls and length replies stay rejected", () => {
  // No database/backend writes: observe the real store helpers through their existing deps.
  const managerStore = mock.method(getSemanticCacheManager(), "store", async () => {});
  try {
    for (const streaming of [false, true]) {
      const entries = new Map<string, unknown>();
      const deps = {
        isCacheableForWrite,
        isSmallEnoughForSemanticCache: () => true,
        generateSignature,
        setCachedResponse: (key: string, _model: string, response: unknown) => {
          entries.set(key, response);
        },
      };
      const body = { messages, temperature: 0, max_tokens: 4096 };
      const complete = {
        choices: [{ finish_reason: "stop", message: { role: "assistant", content: "4" } }],
      };
      const length = {
        choices: [
          {
            finish_reason: "length",
            message: { role: "assistant", content: "", reasoning_content: "working" },
          },
        ],
      };
      const store = (response: typeof complete) =>
        streaming
          ? storeStreamingSemanticCacheResponse(
              {
                enabled: true,
                streamStatus: 200,
                streamResponseBody: response,
                body,
                headers: undefined,
                model: "synthetic-model",
                apiKeyId: "synthetic-key",
              },
              deps
            )
          : storeSemanticCacheResponse(
              {
                enabled: true,
                translatedResponse: response,
                body,
                headers: undefined,
                model: "synthetic-model",
                apiKeyId: "synthetic-key",
              },
              deps
            );
      store(complete);
      assert.deepEqual(entries.get(signature(body)), complete);
      assert.equal(entries.get(signature({ ...body, reasoning_effort: "low" })), undefined);
      assert.equal(entries.get(signature({ ...body, max_tokens: 2048 })), undefined);
      entries.clear();
      store(length);
      assert.equal(entries.size, 0);
    }
  } finally {
    managerStore.mock.restore();
  }
});

test("manager exact identity and controlled vector exclusion survive both response shapes", async () => {
  const { SemanticCacheManager } =
    await import("../../open-sse/services/cache/semanticCacheManager.ts");
  const { MemoryVectorStore } = await import("../../open-sse/services/cache/memoryVectorStore.ts");
  const store = new MemoryVectorStore();
  let embeddings = 0;
  const nearest = mock.method(store, "searchNearest");
  const manager = new SemanticCacheManager(
    { enabled: true, backend: "memory" },
    store,
    async () => {
      embeddings++;
      return { embedding: [1, 0], inputTokens: 1 };
    }
  );
  const base = { messages, temperature: 0 };
  const common = { model: "synthetic-model", provider: "synthetic", apiKeyId: "synthetic-key" };
  try {
    // A real vector entry is present: an incorrect fallback can actually hit it.
    await manager.store({ ...common, body: base, response: { choices: [] } });
    assert.equal(embeddings, 1);
    embeddings = 0;
    for (const finish_reason of ["stop", "length"]) {
      const body = { ...base, max_tokens: 4096, reasoning_effort: "low" };
      const response = {
        choices: [
          {
            finish_reason,
            message: {
              role: "assistant",
              content: finish_reason === "stop" ? "4" : "",
              reasoning_content: "synthetic",
            },
          },
        ],
      };
      await manager.store({ ...common, body, response });
      const same = await manager.lookup({ ...common, body });
      assert.equal(same.type, "exact");
      assert.deepEqual(same.entry?.response, response);
      for (const changed of [{ max_tokens: 2048 }, { reasoning_effort: "high" }]) {
        assert.equal(
          (await manager.lookup({ ...common, body: { ...body, ...changed } })).hit,
          false
        );
      }
      assert.equal((await manager.lookup({ ...common, apiKeyId: "other-key", body })).hit, false);
      assert.equal(embeddings, 0, "controlled store and lookup must not embed");
      assert.equal(nearest.mock.callCount(), 0, "controlled lookup must not search vectors");
      const hash = generateDirectHash(
        common.model,
        messages,
        0,
        1,
        { apiKeyId: common.apiKeyId },
        outputContractOf(body)
      );
      assert.equal((await store.getByHash(hash))?.embedding, undefined);
    }
    const oldHash = crypto
      .createHash("sha256")
      .update(
        JSON.stringify({
          model: common.model,
          provider: "*",
          messages,
          temperature: 0,
          top_p: 1,
        })
      )
      .digest("hex");
    assert.notEqual(generateDirectHash(common.model, messages), oldHash);
    // Old exact entries are ignored; direct mode prevents an unrelated fuzzy hit.
    await store.clear();
    await store.set(
      {
        id: "legacy",
        hash: `${common.apiKeyId}.${oldHash}`,
        promptText: "synthetic",
        ...common,
        response: { legacy: true },
        tokensSaved: 0,
        createdAt: Date.now(),
        expiresAt: Date.now() + 60000,
      },
      60000
    );
    assert.equal(
      (
        await manager.lookup({
          ...common,
          body: base,
          headers: { "x-omniroute-cache-type": "direct" },
        })
      ).hit,
      false
    );
  } finally {
    nearest.mock.restore();
  }
});
