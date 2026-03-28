<template>
  <div class="grid gap-4 lg:grid-cols-3">
    <div class="lg:col-span-2">
      <div class="mb-3 h-[520px] overflow-y-auto rounded border p-3">
        <div v-for="msg in messages" :key="msg.id" class="mb-3">
          <div class="mb-1 text-xs uppercase text-slate-500">{{ msg.role }}</div>
          <div
            class="rounded p-2"
            :class="msg.role === 'assistant' ? 'bg-blue-50' : 'bg-slate-100'"
            v-html="msg.rendered"
          />
        </div>
      </div>

      <form class="flex gap-2" @submit.prevent="sendQuery">
        <input v-model="query" class="flex-1 rounded border px-3 py-2" placeholder="Ask your question..." />
        <button class="rounded bg-blue-600 px-4 py-2 text-white" :disabled="loading">
          {{ loading ? 'Running...' : 'Send' }}
        </button>
      </form>
    </div>

    <div>
      <button class="mb-2 w-full rounded bg-slate-800 px-3 py-2 text-white" @click="thinkingOpen = !thinkingOpen">
        {{ thinkingOpen ? 'Hide' : 'Show' }} Thinking Panel
      </button>
      <div v-if="thinkingOpen" class="h-[520px] overflow-y-auto rounded border p-3">
        <div class="mb-3">
          <h3 class="mb-2 font-semibold">Node Status</h3>
          <ul class="space-y-1 text-sm">
            <li v-for="(item, index) in statuses" :key="`${item.node}-${index}`" class="rounded bg-slate-100 p-2">
              <div class="font-medium">{{ item.node }}</div>
              <div>{{ item.message }}</div>
            </li>
          </ul>
        </div>

        <div class="mb-3">
          <h3 class="mb-2 font-semibold">&lt;think&gt;</h3>
          <div class="rounded bg-amber-50 p-2 text-sm whitespace-pre-wrap">{{ thinkText }}</div>
        </div>

        <div>
          <h3 class="mb-2 font-semibold">&lt;answer&gt;</h3>
          <div class="rounded bg-emerald-50 p-2 text-sm whitespace-pre-wrap">{{ answerText }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { createChatSocket } from '../services/chatSocket'

const sessionId = crypto.randomUUID()
const query = ref('')
const loading = ref(false)
const statuses = ref([])
const rawStream = ref('')
const thinkingOpen = ref(true)

const thinkText = computed(() => {
  const match = rawStream.value.match(/<think>([\s\S]*?)<\/think>/)
  return match ? match[1].trim() : ''
})

const answerText = computed(() => {
  const match = rawStream.value.match(/<answer>([\s\S]*?)<\/answer>/)
  return match ? match[1].trim() : ''
})

const messages = ref([])

function toSafeMarkdownHtml(text) {
  const rawHtml = marked.parse(text)
  return DOMPurify.sanitize(rawHtml)
}

async function sendQuery() {
  const content = query.value.trim()
  if (!content || loading.value) return

  loading.value = true
  statuses.value = []
  rawStream.value = ''

  messages.value.push({
    id: crypto.randomUUID(),
    role: 'user',
    rendered: toSafeMarkdownHtml(content)
  })

  const ws = createChatSocket(sessionId)
  ws.onopen = () => ws.send(JSON.stringify({ query: content }))

  ws.onmessage = (event) => {
    const packet = JSON.parse(event.data)
    if (packet.type === 'status') {
      statuses.value.push({ node: packet.node, message: packet.message })
      return
    }
    if (packet.type === 'token') {
      rawStream.value += packet.content
    }
  }

  ws.onclose = () => {
    const finalAnswer = answerText.value || 'No <answer> tag received.'
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'assistant',
      rendered: toSafeMarkdownHtml(finalAnswer)
    })
    loading.value = false
  }

  ws.onerror = () => {
    loading.value = false
  }

  query.value = ''
}
</script>
