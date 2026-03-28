<template>
  <div class="min-h-screen p-6">
    <div class="mx-auto max-w-6xl rounded-xl bg-white p-4 shadow">
      <div class="mb-4 flex gap-2 border-b pb-3">
        <button
          class="rounded px-4 py-2 text-sm font-medium"
          :class="activeTab === 'chat' ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-700'"
          @click="activeTab = 'chat'"
        >
          Chat QA
        </button>
        <button
          class="rounded px-4 py-2 text-sm font-medium"
          :class="activeTab === 'kb' ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-700'"
          @click="activeTab = 'kb'"
        >
          KB Admin
        </button>
      </div>

      <ChatView v-if="activeTab === 'chat'" />
      <AdminKBView v-else />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ChatView from './views/ChatView.vue'
import AdminKBView from './views/AdminKBView.vue'

const activeTab = ref('chat')
</script>
