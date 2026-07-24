import axios from 'axios'

export const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api/v1').replace(/\/$/, '')

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url, config.data)
    return config
  },
  (error) => {
    console.error('Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.data)
    return response
  },
  (error) => {
    console.error('Response Error:', error.response?.status, error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// API service methods
export const apiService = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health')
    return response.data
  },

  // Get available models
  async getModels() {
    const response = await api.get('/models')
    return response.data
  },

  // Chat with agent
  async chat(messages, model = null, temperature = null, signal = undefined) {
    const payload = { messages }
    if (model) payload.model = model
    if (temperature !== null) payload.temperature = temperature
    
    const response = await api.post('/chat', payload, { signal })  // 对应 /api/v1/chat
    return response.data
  },

  /**
   * Knowledge SSE（POST /knowledge/stream）。handlers: { onMeta, onDelta, onDone, onError }，均为可选。
   */
  async knowledgeQueryStream(payload, handlers = {}, signal = undefined) {
    const body = {
      query: payload.query,
      session_id: payload.session_id ?? 'default',
      ...(payload.model && { model: payload.model }),
      ...(payload.collection != null && { collection: payload.collection }),
      ...(payload.force_multi_doc != null && { force_multi_doc: payload.force_multi_doc }),
      ...(payload.keyword_filter && { keyword_filter: payload.keyword_filter }),
      ...(payload.query_image && { query_image: payload.query_image }),
    }
    const res = await fetch(`${API_BASE}/knowledge/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        // 避免中间层对响应做 gzip，导致整段缓冲后才解压、前端收不到增量
        'Accept-Encoding': 'identity',
      },
      body: JSON.stringify(body),
      signal,
    })
    if (!res.ok) {
      const err = new Error(`stream HTTP ${res.status}`)
      handlers.onError?.(err)
      throw err
    }
    if (!res.body) {
      const err = new Error('stream response body is unavailable')
      handlers.onError?.(err)
      throw err
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')

    const normalizeLf = (s) => s.replace(/\r\n/g, '\n').replace(/\r/g, '\n')

    const dispatchSseBlock = (blockRaw) => {
      const block = blockRaw.trim()
      if (!block) return
      let ev = null
      let dataStr = null
      for (const line of block.split('\n')) {
        const L = line.trimEnd()
        if (L.startsWith('event:')) ev = L.slice(6).trim()
        else if (L.startsWith('data:')) dataStr = L.slice(5).trim()
      }
      if (dataStr == null) return
      let data
      try {
        data = JSON.parse(dataStr)
      } catch (e) {
        handlers.onError?.(e)
        return
      }
      if (ev === 'meta') handlers.onMeta?.(data)
      else if (ev === 'delta') handlers.onDelta?.(data)
      else if (ev === 'done') handlers.onDone?.(data)
      else if (ev === 'error') handlers.onError?.(new Error(data.message || 'stream error'))
    }

    let buffer = ''
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (value) buffer += decoder.decode(value, { stream: !done })
        if (done) {
          buffer += decoder.decode()
          buffer = normalizeLf(buffer)
          let sep
          while ((sep = buffer.indexOf('\n\n')) >= 0) {
            const block = buffer.slice(0, sep)
            buffer = buffer.slice(sep + 2)
            dispatchSseBlock(block)
          }
          if (buffer.trim()) dispatchSseBlock(buffer)
          break
        }
        buffer = normalizeLf(buffer)
        let sep
        while ((sep = buffer.indexOf('\n\n')) >= 0) {
          const block = buffer.slice(0, sep)
          buffer = buffer.slice(sep + 2)
          dispatchSseBlock(block)
        }
      }
    } finally {
      try { await reader.cancel() } catch {}
      reader.releaseLock()
    }
  }
}

export default api