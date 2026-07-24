import { api } from './api'

export const docApi = {
  uploadDocument: (formData) => api.post('/documents/upload', formData),
  uploadDocumentToCategory: (formData) => api.post('/documents/upload-to-category', formData),
  batchUploadToCategory: (formData) => api.post('/documents/batch-upload-to-category', formData),
  searchDocuments: (formData) => api.post('/documents/search', formData),
  startChunking: (categoryId, params = {}) => api.post(`/documents/start-chunking/${categoryId}`, null, { params }),
  getExcelColumns: (categoryFileId) => api.get('/documents/excel-columns', { params: { category_file_id: categoryFileId } }),
  startChunkingExcel: (categoryId, params = {}) => api.post(`/documents/start-chunking-excel/${categoryId}`, null, { params }),

  listJobs: (kbName, limit = 200) => api.get('/jobs', { params: { kb_name: kbName, limit } }),
  getJob: (jobId) => api.get(`/jobs/${encodeURIComponent(jobId)}`),
  upsertJob: (jobId) => api.post(`/jobs/${encodeURIComponent(jobId)}/upsert`),

  getChunksByJob: (jobId) => api.get(`/chunks/job/${encodeURIComponent(jobId)}`),
  editChunk: (jobId, chunkIndex, content) => api.put(`/chunks/job/${encodeURIComponent(jobId)}/chunk/${chunkIndex}`, { content }),
  cleanChunk: (jobId, chunkIndex, instruction) => {
    const fd = new FormData()
    if (instruction) fd.append('instruction', instruction)
    return api.post(`/chunks/job/${encodeURIComponent(jobId)}/chunk/${chunkIndex}/clean`, fd)
  },
  revertChunk: (jobId, chunkIndex) => api.post(`/chunks/job/${encodeURIComponent(jobId)}/chunk/${chunkIndex}/revert`),
  cleanJobChunks: (jobId, instruction) => api.post(`/chunks/job/${encodeURIComponent(jobId)}/clean`, { instruction }),
  revertJobChunks: (jobId) => api.post(`/chunks/job/${encodeURIComponent(jobId)}/revert`),
  upsertJobChunks: (jobId) => api.post(`/chunks/job/${encodeURIComponent(jobId)}/upsert`),
  batchUpsertJobs: (jobIds) => api.post('/chunks/batch-upsert', { job_ids: jobIds }),

  getChunkImages: (jobId, chunkIndex) => api.get(`/chunks/job/${encodeURIComponent(jobId)}/chunk/${chunkIndex}/images`),
  addChunkImage: (jobId, chunkIndex, file, page, insertPosition = 0) => {
    const fd = new FormData()
    fd.append('file', file)
    if (page != null) fd.append('page', page)
    fd.append('insert_position', insertPosition)
    return api.post(`/chunks/job/${encodeURIComponent(jobId)}/chunk/${chunkIndex}/images`, fd)
  },
  deleteChunkImage: (jobId, chunkIndex, imageId) => api.delete(`/chunks/job/${encodeURIComponent(jobId)}/chunk/${chunkIndex}/images/${imageId}`),

  listFiles: (params = {}) => api.get('/files', { params }),
  deleteFile: (fileId) => api.delete('/files', { data: { file_id: fileId } }),
  batchDeleteFiles: (fileIds, kbName) => api.post('/files/batch-delete', { file_ids: fileIds, kb_name: kbName }),

  listCategories: () => api.get('/categories'),
  listCategoryFiles: (categoryId) => api.get(`/categories/${categoryId}`),
  createCategory: (data) => api.post('/categories', data),
  getCategory: (id) => api.get(`/categories/${id}`),
  updateCategory: (id, data) => api.put(`/categories/${id}`, data),
  deleteCategory: (id) => api.delete(`/categories/${id}`),
  deleteCategoryFile: (categoryId, fileId) => api.delete(`/categories/${categoryId}/files/${fileId}`),
  batchDeleteCategoryFiles: (categoryId, fileIds) => api.post(`/categories/${categoryId}/files/batch-delete`, { file_ids: fileIds }),

  listCollections: () => api.get('/admin/collections'),
  createCollection: (data) => api.post('/admin/collections', data),
  updateCollection: (kbName, data) => api.put(`/admin/collections/${kbName}`, data),
  deleteCollection: (kbName) => api.delete(`/admin/collections/${kbName}`),
  getAdminConfig: () => api.get('/admin/config'),

  resolveImages: (placeholders) => api.post('/chunks/resolve-images', { placeholders }),
  resolveQueryImages: (ossKeys) => api.post('/chunks/resolve-oss-keys', { oss_keys: ossKeys }),

  listSessions: (kbName, userId = 'default') => api.get('/conversations', { params: { kb_name: kbName, user_id: userId } }),
  createSession: (kbName, title = '新会话', userId = 'default') => api.post('/conversations', { kb_name: kbName, title, user_id: userId }),
  getSessionMessages: (sessionId, limit = 100) => api.get(`/conversations/${sessionId}/messages`, { params: { limit } }),
  deleteSession: (sessionId) => api.delete(`/conversations/${sessionId}`),
}
