import Anthropic from '@anthropic-ai/sdk'
import {
  MessageCreateParamsNonStreaming,
  MessageCreateParamsStreaming,
} from '@anthropic-ai/sdk/resources/messages'

import { AnthropicCompatibleModel, CompletionParams } from '../chat/index.js'
import {
  CompletionResponse,
  ConfigOptions,
  StreamCompletionResponse,
} from '../userTypes/index.js'
import {
  convertMessages,
  convertStopSequences,
  convertToolParams,
  createCompletionResponseNonStreaming,
  createCompletionResponseStreaming,
  getApiKey,
  getDefaultMaxTokens,
} from './anthropic.js'
import { BaseHandler } from './base.js'
import { InputError } from './types.js'
import { consoleWarn, getTimestamp } from './utils.js'

export class AnthropicCompatibleHandler extends BaseHandler<AnthropicCompatibleModel> {
  constructor(opts: ConfigOptions) {
    // Pass true for all feature flags to allow any model to use any feature
    super(opts, true, true, true, true, true, true)
  }

  protected validateInputs(body: CompletionParams): void {
    // Skip the model validation from BaseHandler by not calling super.validateInputs()
    // Only validate image support for older Claude models
    let logImageDetailWarning: boolean = false
    for (const message of body.messages) {
      if (Array.isArray(message.content)) {
        for (const e of message.content) {
          if (e.type === 'image_url') {
            if (
              e.image_url.detail !== undefined &&
              e.image_url.detail !== 'auto'
            ) {
              logImageDetailWarning = true
            }

            if (
              body.model === 'claude-instant-1.2' ||
              body.model === 'claude-2.0' ||
              body.model === 'claude-2.1'
            ) {
              throw new InputError(
                `Model '${body.model}' does not support images. Remove any images from the prompt or use Claude version 3 or later.`
              )
            }
          }
        }
      }
    }

    if (logImageDetailWarning) {
      consoleWarn(
        `Anthropic does not support the 'detail' field for images. The default image quality will be used.`
      )
    }
  }

  async create(
    body: CompletionParams
  ): Promise<CompletionResponse | StreamCompletionResponse> {
    this.validateInputs(body)

    const apiKey = getApiKey(this.opts.apiKey)
    if (apiKey === undefined) {
      throw new InputError(
        "No Anthropic API key detected. Please define an 'ANTHROPIC_API_KEY' environment variable or supply the API key using the 'apiKey' parameter."
      )
    }

    const stream = typeof body.stream === 'boolean' ? body.stream : undefined
    const maxTokens = body.max_tokens ?? getDefaultMaxTokens(body.model)
    const client = new Anthropic({ apiKey: getApiKey(this.opts.apiKey)! })
    const stopSequences = convertStopSequences(body.stop)
    const topP = typeof body.top_p === 'number' ? body.top_p : undefined
    const temperature =
      typeof body.temperature === 'number'
        ? // We divide by two because Anthropic's temperature range is 0 to 1, unlike OpenAI's, which is
          // 0 to 2.
          body.temperature / 2
        : undefined
    const { messages, systemMessage } = await convertMessages(body.messages)
    const { toolChoice, tools } = convertToolParams(
      body.tool_choice,
      body.tools
    )

    if (stream === true) {
      const convertedBody: MessageCreateParamsStreaming = {
        max_tokens: maxTokens,
        messages,
        model: body.model,
        stop_sequences: stopSequences,
        temperature,
        top_p: topP,
        stream,
        system: systemMessage,
        tools,
        tool_choice: toolChoice,
      }
      const created = getTimestamp()
      const response = client.messages.stream(convertedBody)

      return createCompletionResponseStreaming(response, created)
    } else {
      const convertedBody: MessageCreateParamsNonStreaming = {
        max_tokens: maxTokens,
        messages,
        model: body.model,
        stop_sequences: stopSequences,
        temperature,
        top_p: topP,
        system: systemMessage,
        tools,
        tool_choice: toolChoice,
      }

      const created = getTimestamp()
      const response = await client.messages.create(convertedBody)
      return createCompletionResponseNonStreaming(
        response,
        created,
        body.tool_choice
      )
    }
  }
}
