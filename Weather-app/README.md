# AI Engineer Intern Technical Assessment (React Native Frontend)

This Expo app is the **React Native frontend** for the weather assessment backend.

It covers Tech Assessment #1 requirements while integrating with the existing backend endpoints used for Tech Assessment #2.

## Features

- Search weather by location input (city/zip/landmark/free text)
- Use current GPS location on mobile
- Current weather + 5-day forecast
- Error handling for invalid location/API failures/permission denied
- CRUD UI for weather records (create, read, update, delete)
- Export records (JSON, CSV, Markdown)
- Additional API links (map + YouTube)

## Run

1) Start backend first (from the assessment backend folder):

```bash
cd "../AI Engineering Internship Techincal Assessment/backend"
npm install
npm run dev
```

2) Start this Expo frontend:

```bash
npm install
npx expo start
```

## API base URL notes

The app auto-detects backend URL in most local cases:

- Android emulator: `http://10.0.2.2:4000/api`
- iOS simulator/web: host machine `:4000`

You can override manually:

```bash
EXPO_PUBLIC_API_BASE=http://YOUR_IP:4000/api npx expo start
```

Use your LAN IP when testing on a physical phone.
