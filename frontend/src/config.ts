const backendOrigin = (import.meta.env.VITE_BACKEND_ORIGIN ?? '').replace(/\/$/, '')

export const BACKEND_ORIGIN = backendOrigin
export const API_URL = `${backendOrigin}/api`
