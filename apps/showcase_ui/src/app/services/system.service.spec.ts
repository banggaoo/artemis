import { provideHttpClient, withXhr } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { SystemReadinessReport } from '../core/models/system.model';
import { SystemService } from './system.service';

describe('SystemService readiness polling', () => {
  let service: SystemService;
  let http: HttpTestingController;

  const report = (timestamp: number): SystemReadinessReport => ({
    overall_ready: true,
    blocker_count: 3,
    passed_blocker_count: 3,
    probes: [],
    active_device: null,
    os_type: 'windows',
    timestamp
  });

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        SystemService,
        provideHttpClient(withXhr()),
        provideHttpClientTesting()
      ]
    });
    service = TestBed.inject(SystemService);
    http = TestBed.inject(HttpTestingController);
    service.stopAutoPolling();
  });

  afterEach(() => http.verify());

  it('shares one HTTP request across overlapping readiness callers', () => {
    let firstTimestamp = 0;
    let secondTimestamp = 0;

    service.fetchReadiness().subscribe(value => firstTimestamp = value.timestamp);
    service.fetchReadiness(true).subscribe(value => secondTimestamp = value.timestamp);

    const request = http.expectOne('/api/system/readiness');
    request.flush(report(10));

    expect(firstTimestamp).toBe(10);
    expect(secondTimestamp).toBe(10);
    expect(service.readinessReport()?.timestamp).toBe(10);
    expect(service.isLoading()).toBeFalse();
  });

  it('does not let an older snapshot replace a newer report', () => {
    service.fetchReadiness().subscribe();
    http.expectOne('/api/system/readiness').flush(report(20));

    service.fetchReadiness().subscribe();
    http.expectOne('/api/system/readiness').flush(report(10));

    expect(service.readinessReport()?.timestamp).toBe(20);
  });

  it('requests a forced backend refresh for a manual re-check', () => {
    service.fetchReadiness(false, true).subscribe();

    const request = http.expectOne(req =>
      req.url === '/api/system/readiness' && req.params.get('force') === 'true'
    );
    expect(request.request.params.get('force')).toBe('true');
    request.flush(report(30));
  });

  it('loads the active ADB server endpoint', () => {
    service.fetchAdbServerStatus().subscribe();

    const request = http.expectOne('/api/system/adb/server');
    request.flush({
      endpoint: {
        host: '127.0.0.1',
        port: 5038,
        socket: 'tcp:127.0.0.1:5038',
        mode: 'remote',
        is_local_default: false
      }
    });

    expect(service.isRemoteAdbServer()).toBeTrue();
    expect(service.adbServerStatus()?.endpoint.port).toBe(5038);
  });

  it('activates an ADB server endpoint and applies its readiness report', () => {
    service.connectAdbServer('127.0.0.1', 5038, true).subscribe();

    const request = http.expectOne('/api/system/adb/server/connect');
    expect(request.request.body).toEqual({
      host: '127.0.0.1',
      port: 5038,
      persist: true
    });
    request.flush({
      connection_result: {
        success: true,
        message: 'Connected.',
        endpoint: {
          host: '127.0.0.1',
          port: 5038,
          socket: 'tcp:127.0.0.1:5038',
          mode: 'remote',
          is_local_default: false
        },
        devices: []
      },
      report: report(40)
    });

    expect(service.isRemoteAdbServer()).toBeTrue();
    expect(service.readinessReport()?.timestamp).toBe(40);
  });

  it('probes an ADB server without changing the active endpoint', () => {
    service.probeAdbServer('127.0.0.1', 5038).subscribe();

    const request = http.expectOne('/api/system/adb/server/probe');
    expect(request.request.body).toEqual({
      host: '127.0.0.1',
      port: 5038,
      persist: false
    });
    request.flush({
      connection_result: {
        success: true,
        message: 'Endpoint is reachable.',
        endpoint: {
          host: '127.0.0.1',
          port: 5038,
          socket: 'tcp:127.0.0.1:5038',
          identity: 'tcp:127.0.0.1:5038',
          mode: 'remote',
          is_local_default: false
        },
        devices: []
      }
    });

    expect(service.adbServerStatus()).toBeNull();
  });

  it('merges iOS simulators into the device list with platform tags', () => {
    service.fetchReadiness().subscribe();
    http.expectOne('/api/system/readiness').flush({
      ...report(60),
      probes: [
        {
          id: 'android_adb',
          category: 'device',
          title: 'Android Device',
          status: 'pass',
          is_blocker: true,
          summary: 'Device Ready',
          description: '',
          metadata: {
            devices: [
              {
                serial: 'emulator-5554',
                state: 'device',
                model: 'Pixel_8',
                product: 'sdk',
                android_version: '14',
                screen_resolution: '1080x2400',
                is_locked: false,
                is_emulator: true
              }
            ]
          },
          actions: []
        },
        {
          id: 'ios_simulators',
          category: 'device',
          title: 'iOS Simulator',
          status: 'pass',
          is_blocker: false,
          summary: '1 Booted',
          description: '',
          metadata: {
            simulators: [
              {
                udid: 'E1D9F1D1-04E5-4E95-801F-830B854FD3E2',
                name: 'iPhone 18 Pro',
                state: 'Booted',
                runtime: 'com.apple.CoreSimulator.SimRuntime.iOS-27-0',
                isAvailable: true
              },
              {
                udid: 'AAAA1111-2222-3333-4444-555566667777',
                name: 'iPhone 17',
                state: 'Shutdown',
                runtime: 'com.apple.CoreSimulator.SimRuntime.iOS-26-0',
                isAvailable: true
              }
            ]
          },
          actions: []
        }
      ]
    });

    const devices = service.connectedDevices();
    expect(devices.length).toBe(3);
    expect(devices[0].platform).toBe('android');
    const booted = devices.find(d => d.serial === 'E1D9F1D1-04E5-4E95-801F-830B854FD3E2');
    expect(booted?.platform).toBe('ios');
    expect(booted?.model).toBe('iPhone 18 Pro');
    expect(booted?.state).toBe('device');
    expect(booted?.product).toBe('iOS 27.0');
    expect(booted?.is_emulator).toBeTrue();
    const shutdown = devices.find(d => d.serial === 'AAAA1111-2222-3333-4444-555566667777');
    expect(shutdown?.state).toBe('Shutdown');
  });

  it('selects an iOS simulator by posting its platform and tracks it locally', () => {
    service.fetchReadiness().subscribe();
    http.expectOne('/api/system/readiness').flush({
      ...report(70),
      probes: [
        {
          id: 'ios_simulators',
          category: 'device',
          title: 'iOS Simulator',
          status: 'pass',
          is_blocker: false,
          summary: '1 Booted',
          description: '',
          metadata: {
            simulators: [
              {
                udid: 'UDID-1',
                name: 'iPhone 18 Pro',
                state: 'Booted',
                runtime: 'com.apple.CoreSimulator.SimRuntime.iOS-27-0',
                isAvailable: true
              }
            ]
          },
          actions: []
        }
      ]
    });

    service.selectDevice('UDID-1', 'ios').subscribe();
    const request = http.expectOne('/api/system/devices/select');
    expect(request.request.body).toEqual({ serial: 'UDID-1', platform: 'ios' });
    request.flush({ status: 'success', selected_serial: 'UDID-1', platform: 'ios' });

    expect(service.selectedIosDevice()?.serial).toBe('UDID-1');
    expect(service.selectedDeviceSerial()).toBe('UDID-1');
    expect(service.isDeviceReady()).toBeTrue();
  });

  it('clears the iOS selection when an Android device is selected instead', () => {
    service.selectedIosDevice.set({
      serial: 'UDID-1',
      state: 'device',
      model: 'iPhone',
      product: 'iOS 27.0',
      android_version: null,
      screen_resolution: null,
      is_locked: null,
      is_emulator: true,
      platform: 'ios'
    });

    service.selectDevice('emulator-5554', 'android').subscribe();
    const request = http.expectOne('/api/system/devices/select');
    expect(request.request.body).toEqual({ serial: 'emulator-5554', platform: 'android' });
    request.flush({ status: 'success', selected_serial: 'emulator-5554', report: report(80) });

    expect(service.selectedIosDevice()).toBeNull();
  });

  it('restores the standard local ADB server explicitly', () => {
    service.useLocalAdbServer(true).subscribe();

    const request = http.expectOne(req =>
      req.url === '/api/system/adb/server/local' && req.params.get('persist') === 'true'
    );
    request.flush({
      connection_result: {
        success: true,
        message: 'Using local ADB.',
        endpoint: {
          host: '127.0.0.1',
          port: 5037,
          socket: 'tcp:127.0.0.1:5037',
          mode: 'local',
          is_local_default: true
        },
        devices: []
      },
      report: report(50)
    });

    expect(service.isRemoteAdbServer()).toBeFalse();
    expect(service.adbServerStatus()?.endpoint.port).toBe(5037);
  });
});
