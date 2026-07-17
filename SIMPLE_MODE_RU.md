# Упрощённый режим: русские клипы без публикации

Этот профиль оставляет только основной сценарий:

1. загрузить длинное видео;
2. распознать русскую речь;
3. выбрать лучшие фрагменты через LLM;
4. собрать вертикальные клипы 9:16;
5. наложить синхронные субтитры;
6. скачать готовые MP4.

Instagram, Stripe, Resend, Discord, Pexels и автоматическая публикация в этом режиме отключены через пустые переменные окружения. Backend, Redis и PostgreSQL не публикуются наружу.

## Подготовка

```bash
cp .env.simple.example .env
```

Заполните минимум:

- `BETTER_AUTH_SECRET`;
- `BACKEND_AUTH_SECRET`;
- `POSTGRES_PASSWORD`;
- `REDIS_PASSWORD`;
- `ASSEMBLY_AI_API_KEY`;
- ключ выбранной LLM, например `GOOGLE_API_KEY`.

Секреты можно создать так:

```bash
openssl rand -hex 32
```

## Запуск

```bash
docker compose -f docker-compose.yml -f docker-compose.simple.yml up -d --build
```

Откройте:

```text
http://localhost:3000
```

## Настройки для VPS 2 vCPU / 4 GB RAM

Профиль ограничивает ресурсы приблизительно так:

- worker: 1.5 vCPU и 2.3 GB RAM;
- backend: 0.5 vCPU и 512 MB RAM;
- PostgreSQL: 0.3 vCPU и 384 MB RAM;
- Redis: 0.2 vCPU и 160 MB RAM.

Рекомендуется добавить 4 GB swap и обрабатывать только одно видео одновременно.

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Что пока остаётся

Текущий надёжный путь транскрибации использует AssemblyAI. Переменная `TRANSCRIPT_LANGUAGE=ru` передаётся в backend и worker, но фактическое поведение зависит от текущей реализации транскрибатора.

Следующий этап упрощения:

- добавить CPU-совместимый `faster-whisper`;
- сделать локальную транскрибацию основным режимом;
- скрыть из интерфейса тарифы, Instagram и остальные SaaS-разделы;
- добавить одну страницу «Загрузить → Обработать → Скачать»;
- добавить скачивание всех клипов ZIP-архивом.

## Остановка

```bash
docker compose -f docker-compose.yml -f docker-compose.simple.yml down
```
