<template>
  <div class="space-y-4">
    <div
      class="rounded border-2 border-dashed border-slate-300 p-6 text-center"
      @dragover.prevent
      @drop.prevent="onDrop"
    >
      <p class="mb-2 font-medium">Drag & drop a file here (PDF/TXT/MD)</p>
      <input ref="fileInput" type="file" class="hidden" @change="onFileInput" />
      <button class="rounded bg-blue-600 px-4 py-2 text-white" @click="fileInput.click()">Choose File</button>
      <div v-if="uploading" class="mt-3">
        <div class="h-2 w-full rounded bg-slate-200">
          <div class="h-2 rounded bg-blue-600" :style="{ width: `${progress}%` }" />
        </div>
        <p class="mt-1 text-sm">Uploading... {{ progress }}%</p>
      </div>
    </div>

    <div class="rounded border">
      <table class="w-full text-sm">
        <thead class="bg-slate-100 text-left">
          <tr>
            <th class="p-2">File ID</th>
            <th class="p-2">Name</th>
            <th class="p-2">Status</th>
            <th class="p-2">Chunks</th>
            <th class="p-2">Created</th>
            <th class="p-2">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in files" :key="item.file_id" class="border-t">
            <td class="p-2 text-xs">{{ item.file_id }}</td>
            <td class="p-2">{{ item.file_name }}</td>
            <td class="p-2">{{ item.status }}</td>
            <td class="p-2">{{ item.chunk_count }}</td>
            <td class="p-2">{{ item.created_at }}</td>
            <td class="p-2">
              <button class="rounded bg-red-600 px-3 py-1 text-white" @click="remove(item.file_id)">Delete</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { deleteFile, listFiles, uploadFile } from '../services/api'

const fileInput = ref(null)
const files = ref([])
const uploading = ref(false)
const progress = ref(0)

async function refresh() {
  const response = await listFiles(100, 0)
  files.value = response.items
}

async function upload(selected) {
  uploading.value = true
  progress.value = 20
  await uploadFile(selected)
  progress.value = 100
  await refresh()
  setTimeout(() => {
    uploading.value = false
    progress.value = 0
  }, 400)
}

async function onFileInput(event) {
  const selected = event.target.files?.[0]
  if (!selected) return
  await upload(selected)
  event.target.value = ''
}

async function onDrop(event) {
  const selected = event.dataTransfer?.files?.[0]
  if (!selected) return
  await upload(selected)
}

async function remove(fileId) {
  await deleteFile(fileId)
  await refresh()
}

onMounted(refresh)
</script>
