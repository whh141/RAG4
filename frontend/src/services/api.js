import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
})

export async function uploadFile(file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await api.post('/api/kb/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return response.data
}

export async function listFiles(limit = 20, offset = 0) {
  const response = await api.get('/api/kb/files', { params: { limit, offset } })
  return response.data
}

export async function deleteFile(fileId) {
  const response = await api.delete(`/api/kb/files/${fileId}`)
  return response.data
}
