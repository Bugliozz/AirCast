-- Air Quality DB — MySQL 8.0
-- Auto-executed by Docker on first container start.

CREATE DATABASE IF NOT EXISTS airquality
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE airquality;

-- ──────────────────────────────────────────────
-- Monitoring stations (anagrafica stazioni)
-- ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stations (
    idstazione    VARCHAR(20)   NOT NULL,
    nomestazione  VARCHAR(150),
    provincia     VARCHAR(50),
    comune        VARCHAR(100),
    zona          VARCHAR(50),   -- urbana / suburbana / rurale / industriale
    quota         SMALLINT,      -- metres above sea level
    lat           DECIMAL(10,7),
    lng           DECIMAL(10,7),
    PRIMARY KEY (idstazione)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ──────────────────────────────────────────────
-- Sensors (anagrafica sensori)
-- ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sensors (
    idsensore     VARCHAR(20)   NOT NULL,
    idstazione    VARCHAR(20),
    tiposensore   VARCHAR(50),   -- PM10, PM2.5, NO2, O3, CO …
    unitamisura   VARCHAR(20),   -- µg/m³, mg/m³ …
    datastart     DATE,
    datastop      DATE,          -- NULL → still active
    PRIMARY KEY (idsensore),
    KEY idx_sensors_stazione (idstazione)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ──────────────────────────────────────────────
-- Measurements  (misure orarie / giornaliere)
-- PM10 / PM2.5 → one record per day (T00:00:00)
-- NO2 / O3 / CO → one record per hour
-- ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS measurements (
    id            INT UNSIGNED  NOT NULL AUTO_INCREMENT,
    idsensore     VARCHAR(20)   NOT NULL,
    data          DATETIME      NOT NULL,
    valore        FLOAT,
    stato         VARCHAR(5),
    PRIMARY KEY (id),
    UNIQUE KEY uq_meas        (idsensore, data),
    KEY          idx_meas_data    (data),
    KEY          idx_meas_sensore (idsensore)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ──────────────────────────────────────────────
-- Hourly weather per station  (Open-Meteo)
-- ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS weather_hourly (
    id                    INT UNSIGNED  NOT NULL AUTO_INCREMENT,
    idstazione            VARCHAR(20)   NOT NULL,
    dt                    DATETIME      NOT NULL,
    temperature_2m        FLOAT,
    relative_humidity_2m  FLOAT,
    dew_point_2m          FLOAT,
    precipitation         FLOAT,
    surface_pressure      FLOAT,
    cloud_cover           FLOAT,
    wind_speed_10m        FLOAT,
    wind_direction_10m    FLOAT,
    visibility            FLOAT,
    shortwave_radiation   FLOAT,
    boundary_layer_height FLOAT,
    PRIMARY KEY (id),
    UNIQUE KEY uq_weather          (idstazione, dt),
    KEY        idx_weather_dt      (dt),
    KEY        idx_weather_stazione (idstazione)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
