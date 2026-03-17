import Constants from "expo-constants";
import { Platform } from "react-native";

function resolveBaseUrl() {
  const env = process.env.EXPO_PUBLIC_API_BASE;
  if (env) return env;

  if (Platform.OS === "android") {
    return "http://10.0.2.2:4000/api";
  }

  const hostUri =
    (Constants.expoConfig as { hostUri?: string } | null)?.hostUri ??
    (Constants as unknown as { manifest2?: { extra?: { expoClient?: { hostUri?: string } } } }).manifest2
      ?.extra?.expoClient?.hostUri;

  if (hostUri) {
    const host = hostUri.split(":")[0];
    return `http://${host}:4000/api`;
  }

  return "http://localhost:4000/api";
}

export const API_BASE = resolveBaseUrl();

export async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
    ...options
  });
  if (!res.ok) {
    let message = "Request failed.";
    try {
      const body = (await res.json()) as { error?: string };
      if (body?.error) message = body.error;
    } catch {
      // no-op
    }
    throw new Error(message);
  }

  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}
