import React, { useEffect, useMemo, useState } from "react";
import { Linking, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { apiFetch, API_BASE } from "@/lib/api";

type RecordItem = {
  id: number;
  location_input: string;
  normalized_location: string;
  latitude: number;
  longitude: number;
  start_date: string;
  end_date: string;
};

export default function RecordsScreen() {
  const [records, setRecords] = useState<RecordItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({
    location: "Chicago",
    startDate: "",
    endDate: ""
  });
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editDraft, setEditDraft] = useState({
    location: "",
    startDate: "",
    endDate: ""
  });

  const today = useMemo(() => new Date().toISOString().slice(0, 10), []);

  useEffect(() => {
    const end = new Date();
    const start = new Date();
    start.setDate(end.getDate() - 6);
    setForm((prev) => ({
      ...prev,
      startDate: start.toISOString().slice(0, 10),
      endDate: end.toISOString().slice(0, 10)
    }));
    void loadRecords();
  }, []);

  async function loadRecords() {
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch<RecordItem[]>("/requests");
      setRecords(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to load records.");
    } finally {
      setLoading(false);
    }
  }

  async function createRecord() {
    setError(null);
    setLoading(true);
    try {
      await apiFetch("/requests", {
        method: "POST",
        body: JSON.stringify(form)
      });
      await loadRecords();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to create record.");
    } finally {
      setLoading(false);
    }
  }

  function beginEdit(item: RecordItem) {
    setEditingId(item.id);
    setEditDraft({
      location: item.location_input,
      startDate: item.start_date,
      endDate: item.end_date
    });
  }

  async function saveEdit(id: number) {
    setError(null);
    setLoading(true);
    try {
      await apiFetch(`/requests/${id}`, {
        method: "PUT",
        body: JSON.stringify(editDraft)
      });
      setEditingId(null);
      await loadRecords();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to update record.");
    } finally {
      setLoading(false);
    }
  }

  async function deleteRecord(id: number) {
    setError(null);
    setLoading(true);
    try {
      await apiFetch(`/requests/${id}`, { method: "DELETE" });
      await loadRecords();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to delete record.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Records (CRUD)</Text>
      <Text style={styles.subtitle}>Create, read, update, delete + export</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Create Record</Text>
        <TextInput
          style={styles.input}
          placeholder="Location"
          placeholderTextColor="#9CA3AF"
          value={form.location}
          onChangeText={(v) => setForm((p) => ({ ...p, location: v }))}
        />
        <View style={styles.row}>
          <TextInput
            style={[styles.input, styles.dateInput]}
            value={form.startDate}
            onChangeText={(v) => setForm((p) => ({ ...p, startDate: v }))}
            placeholder="YYYY-MM-DD"
            placeholderTextColor="#9CA3AF"
          />
          <TextInput
            style={[styles.input, styles.dateInput]}
            value={form.endDate}
            onChangeText={(v) => setForm((p) => ({ ...p, endDate: v }))}
            placeholder="YYYY-MM-DD"
            placeholderTextColor="#9CA3AF"
          />
        </View>
        <Pressable style={styles.button} onPress={createRecord} disabled={loading}>
          <Text style={styles.buttonText}>Create</Text>
        </Pressable>
        <Text style={styles.smallText}>Use date format YYYY-MM-DD, up to today {today}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Export</Text>
        <View style={styles.row}>
          <Pressable style={styles.secondaryButton} onPress={() => Linking.openURL(`${API_BASE}/exports/json`)}>
            <Text style={styles.buttonText}>JSON</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => Linking.openURL(`${API_BASE}/exports/csv`)}>
            <Text style={styles.buttonText}>CSV</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => Linking.openURL(`${API_BASE}/exports/md`)}>
            <Text style={styles.buttonText}>Markdown</Text>
          </Pressable>
        </View>
      </View>

      {error ? <Text style={styles.error}>{error}</Text> : null}
      <Pressable style={styles.secondaryButton} onPress={loadRecords}>
        <Text style={styles.buttonText}>Refresh</Text>
      </Pressable>

      {records.map((item) => (
        <View key={item.id} style={styles.recordCard}>
          <Text style={styles.recordTitle}>#{item.id} - {item.normalized_location}</Text>
          <Text style={styles.smallText}>
            Coords: {item.latitude.toFixed(3)}, {item.longitude.toFixed(3)}
          </Text>

          {editingId === item.id ? (
            <>
              <TextInput
                style={styles.input}
                value={editDraft.location}
                onChangeText={(v) => setEditDraft((p) => ({ ...p, location: v }))}
              />
              <View style={styles.row}>
                <TextInput
                  style={[styles.input, styles.dateInput]}
                  value={editDraft.startDate}
                  onChangeText={(v) => setEditDraft((p) => ({ ...p, startDate: v }))}
                />
                <TextInput
                  style={[styles.input, styles.dateInput]}
                  value={editDraft.endDate}
                  onChangeText={(v) => setEditDraft((p) => ({ ...p, endDate: v }))}
                />
              </View>
              <View style={styles.row}>
                <Pressable style={styles.button} onPress={() => saveEdit(item.id)}>
                  <Text style={styles.buttonText}>Save</Text>
                </Pressable>
                <Pressable style={styles.secondaryButton} onPress={() => setEditingId(null)}>
                  <Text style={styles.buttonText}>Cancel</Text>
                </Pressable>
              </View>
            </>
          ) : (
            <>
              <Text style={styles.smallText}>
                Range: {item.start_date} to {item.end_date}
              </Text>
              <View style={styles.row}>
                <Pressable style={styles.secondaryButton} onPress={() => beginEdit(item)}>
                  <Text style={styles.buttonText}>Update</Text>
                </Pressable>
                <Pressable style={styles.dangerButton} onPress={() => deleteRecord(item.id)}>
                  <Text style={styles.buttonText}>Delete</Text>
                </Pressable>
              </View>
            </>
          )}
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0B1220" },
  content: { padding: 16, gap: 12, paddingBottom: 30 },
  title: { color: "#F9FAFB", fontSize: 24, fontWeight: "700" },
  subtitle: { color: "#94A3B8" },
  card: {
    backgroundColor: "#111827",
    borderWidth: 1,
    borderColor: "#1F2937",
    borderRadius: 12,
    padding: 12,
    gap: 8
  },
  cardTitle: { color: "#F9FAFB", fontWeight: "700" },
  input: {
    backgroundColor: "#1E293B",
    color: "#F9FAFB",
    borderWidth: 1,
    borderColor: "#334155",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10
  },
  row: { flexDirection: "row", gap: 8, flexWrap: "wrap" },
  dateInput: { flex: 1, minWidth: 120 },
  button: {
    backgroundColor: "#2563EB",
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 14
  },
  secondaryButton: {
    backgroundColor: "#475569",
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 14
  },
  dangerButton: {
    backgroundColor: "#B91C1C",
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 14
  },
  buttonText: { color: "#F9FAFB", fontWeight: "600" },
  error: { color: "#FCA5A5", fontWeight: "600" },
  smallText: { color: "#CBD5E1" },
  recordCard: {
    backgroundColor: "#0F172A",
    borderWidth: 1,
    borderColor: "#334155",
    borderRadius: 12,
    padding: 10,
    gap: 6
  },
  recordTitle: { color: "#E2E8F0", fontWeight: "700" }
});
