import * as Location from "expo-location";
import React, { useMemo, useState } from "react";
import { ActivityIndicator, Linking, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { apiFetch } from "@/lib/api";

type WeatherResponse = {
  location: string;
  latitude: number;
  longitude: number;
  current: {
    temperature: number;
    apparentTemperature: number;
    humidity: number;
    weatherCode: number;
    weatherLabel: string;
    windSpeed: number;
  };
  forecast: {
    date: string;
    weatherCode: number;
    weatherLabel: string;
    maxTemp: number;
    minTemp: number;
    precipProbability: number;
  }[];
  extras: {
    mapUrl: string;
    youtubeSearchUrl: string;
    wikiSummary: string | null;
  };
};

function iconForCode(code: number) {
  if (code === 0) return "☀️";
  if ([1, 2, 3].includes(code)) return "⛅";
  if ([45, 48].includes(code)) return "🌫️";
  if ([51, 53, 55, 61, 63, 65, 80, 81, 82].includes(code)) return "🌧️";
  if ([71, 73, 75].includes(code)) return "❄️";
  if (code >= 95) return "⛈️";
  return "🌤️";
}

export default function WeatherScreen() {
  const [query, setQuery] = useState("Chicago");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [weather, setWeather] = useState<WeatherResponse | null>(null);
  const today = useMemo(() => new Date().toISOString().slice(0, 10), []);

  async function searchByLocation() {
    setError(null);
    setLoading(true);
    try {
      const data = await apiFetch<WeatherResponse>("/weather/search", {
        method: "POST",
        body: JSON.stringify({ location: query })
      });
      setWeather(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch weather.");
    } finally {
      setLoading(false);
    }
  }

  async function searchByCurrentLocation() {
    setError(null);
    setLoading(true);
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== "granted") {
        throw new Error("Location permission denied.");
      }
      const position = await Location.getCurrentPositionAsync({});
      const data = await apiFetch<WeatherResponse>("/weather/search", {
        method: "POST",
        body: JSON.stringify({
          lat: position.coords.latitude,
          lon: position.coords.longitude
        })
      });
      setWeather(data);
      setQuery(data.location);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not get current location weather.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Weather Search</Text>
      <Text style={styles.subtitle}>Current weather + 5-day forecast (real API data)</Text>

      <TextInput
        style={styles.input}
        placeholder="City, zip, landmark, etc."
        placeholderTextColor="#9CA3AF"
        value={query}
        onChangeText={setQuery}
      />

      <View style={styles.row}>
        <Pressable style={styles.button} onPress={searchByLocation} disabled={loading}>
          <Text style={styles.buttonText}>Search</Text>
        </Pressable>
        <Pressable style={[styles.button, styles.secondaryButton]} onPress={searchByCurrentLocation} disabled={loading}>
          <Text style={styles.buttonText}>Use Current Location</Text>
        </Pressable>
      </View>

      {loading && <ActivityIndicator />}
      {error ? <Text style={styles.error}>{error}</Text> : null}

      {weather ? (
        <View style={styles.card}>
          <Text style={styles.location}>{weather.location}</Text>
          <Text style={styles.temp}>
            {iconForCode(weather.current.weatherCode)} {weather.current.temperature}°C
          </Text>
          <Text style={styles.info}>Feels like: {weather.current.apparentTemperature}°C</Text>
          <Text style={styles.info}>Humidity: {weather.current.humidity}%</Text>
          <Text style={styles.info}>Wind: {weather.current.windSpeed} km/h</Text>
          <Text style={styles.info}>Condition: {weather.current.weatherLabel}</Text>
        </View>
      ) : null}

      {weather ? (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>5-Day Forecast</Text>
          {weather.forecast.map((d) => (
            <View key={d.date} style={styles.forecastRow}>
              <Text style={styles.forecastDate}>{d.date}</Text>
              <Text>
                {iconForCode(d.weatherCode)} {d.weatherLabel}
              </Text>
              <Text>
                {d.minTemp}° / {d.maxTemp}°
              </Text>
              <Text>Rain: {d.precipProbability ?? 0}%</Text>
            </View>
          ))}
        </View>
      ) : null}

      {weather ? (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Extra APIs</Text>
          <Pressable onPress={() => Linking.openURL(weather.extras.mapUrl)}>
            <Text style={styles.link}>Open map location</Text>
          </Pressable>
          <Pressable onPress={() => Linking.openURL(weather.extras.youtubeSearchUrl)}>
            <Text style={styles.link}>Open YouTube location videos</Text>
          </Pressable>
          {weather.extras.wikiSummary ? <Text style={styles.info}>{weather.extras.wikiSummary}</Text> : null}
        </View>
      ) : null}

      <Text style={styles.footer}>Today: {today}</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0B1220" },
  content: { padding: 16, gap: 12, paddingBottom: 24 },
  title: { color: "#F9FAFB", fontSize: 24, fontWeight: "700" },
  subtitle: { color: "#94A3B8" },
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
  button: {
    backgroundColor: "#2563EB",
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 14
  },
  secondaryButton: { backgroundColor: "#475569" },
  buttonText: { color: "#F9FAFB", fontWeight: "600" },
  error: { color: "#FCA5A5", fontWeight: "600" },
  card: {
    backgroundColor: "#111827",
    borderWidth: 1,
    borderColor: "#1F2937",
    borderRadius: 12,
    padding: 12,
    gap: 6
  },
  location: { color: "#E2E8F0", fontWeight: "600", fontSize: 17 },
  temp: { color: "#F9FAFB", fontWeight: "700", fontSize: 28 },
  info: { color: "#CBD5E1" },
  cardTitle: { color: "#F9FAFB", fontWeight: "700", marginBottom: 6 },
  forecastRow: {
    borderWidth: 1,
    borderColor: "#374151",
    borderRadius: 10,
    padding: 8,
    gap: 2
  },
  forecastDate: { color: "#FBBF24", fontWeight: "600" },
  link: { color: "#60A5FA", textDecorationLine: "underline", marginBottom: 4 },
  footer: { color: "#64748B", textAlign: "center", marginTop: 8 }
});
